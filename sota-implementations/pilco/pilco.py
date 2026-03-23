import hydra
import tensordict
import torch
from omegaconf import DictConfig

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl._utils import get_available_device, logger as torchrl_logger, timeit
from torchrl.data import Composite, Unbounded
from torchrl.envs import TransformedEnv
from torchrl.envs.custom.pendulum import PendulumEnv
from torchrl.envs.model_based import ImaginedEnv
from torchrl.envs.transforms import MeanActionSelector
from torchrl.envs.utils import RandomPolicy
from torchrl.modules.models import GPWorldModel, RBFController
from torchrl.objectives import ExponentialQuadraticCost
from torchrl.record.loggers import generate_exp_name, get_logger, Logger

from utils import make_env


def eval_policy(eval_env, policy_module, reset_td, max_steps=200):
    """Runs one eval rollout from a fixed starting state."""
    td = eval_env.rollout(
        max_steps=max_steps,
        policy=policy_module,
        tensordict=eval_env.reset(reset_td.clone()),
        break_when_any_done=True,
    )
    return td, {
        "eval/episode_reward": td["next", "episode_reward"][-1].item(),
        "eval/steps": td["next", "step_count"][-1].item(),
    }


def flatten_eval_rollout(test_rollout):
    """Extracts flat observation/action tensors from a MeanActionSelector rollout."""
    td = test_rollout.exclude("observation", "action", ("next", "observation"))
    td["observation"] = test_rollout["observation", "mean"]
    td["action"] = test_rollout["action", "mean"]
    td[("next", "observation")] = test_rollout["next", "observation", "mean"]
    return td


def _save_policy_state(policy_module):
    return {n: p.data.clone() for n, p in policy_module.named_parameters()}


def _restore_policy_state(policy_module, state):
    for n, p in policy_module.named_parameters():
        p.data.copy_(state[n])


def pilco_loop(
    cfg: DictConfig, env: TransformedEnv, logger: Logger | None = None
) -> TensorDictModule:
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    dtype = torch.float64

    # Fixed starting state: pendulum hanging down (th=π, thdot=0)
    reset_td = PendulumEnv.gen_params(device=env.device)
    reset_td["th"] = torch.tensor(torch.pi, device=env.device)
    reset_td["thdot"] = torch.tensor(0.0, device=env.device)

    # Collect initial data with random policy from fixed start
    random_policy = RandomPolicy(action_spec=env.action_spec)
    initial_rollout = env.rollout(
        max_steps=cfg.pilco.initial_rollout_length,
        policy=random_policy,
        tensordict=env.reset(reset_td.clone()),
    )
    n_initial = len(initial_rollout)
    rollout = initial_rollout.clone()

    n_basis = cfg.pilco.policy_n_basis
    base_policy = (
        RBFController(
            input_dim=obs_dim,
            output_dim=action_dim,
            n_basis=n_basis,
            max_action=env.action_spec.high,
        )
        .to(env.device)
        .to(dtype)
    )
    indices = torch.randperm(len(rollout))[:n_basis]
    base_policy.centers.data.copy_(rollout["observation"][indices].to(dtype))
    policy_module = TensorDictModule(
        module=base_policy,
        in_keys=[("observation", "mean"), ("observation", "var")],
        out_keys=[
            ("action", "mean"),
            ("action", "var"),
            ("action", "cross_covariance"),
        ],
    )

    initial_state_var = cfg.pilco.get("initial_state_var", 1e-4)
    initial_observation = TensorDict(
        {
            ("observation", "mean"): rollout["observation"][0].to(dtype=dtype),
            ("observation", "var"): torch.eye(obs_dim, device=env.device, dtype=dtype)
            * initial_state_var,
        }
    )

    eval_env = TransformedEnv(env, MeanActionSelector())

    target = torch.tensor(cfg.pilco.target, dtype=dtype, device=env.device)
    weights = torch.tensor(cfg.pilco.weights, dtype=dtype, device=env.device)
    cost_module = ExponentialQuadraticCost(
        target=target, weights=weights, reduction="none"
    ).to(env.device)

    eval_max_steps = cfg.pilco.eval_max_steps

    # Baseline eval
    with torch.no_grad():
        baseline_rollout, baseline_metrics = eval_policy(
            eval_env, policy_module, reset_td, eval_max_steps
        )
        baseline_cost = cost_module(baseline_rollout).get("loss_cost").sum().item()
        baseline_metrics["eval/pilco_cost"] = baseline_cost
    torchrl_logger.info(
        f"Baseline | reward: {baseline_metrics['eval/episode_reward']:.2f} | "
        f"pilco cost: {baseline_cost:.2f}",
    )
    if logger:
        logger.log_metrics(baseline_metrics, 0)

    best_reward = baseline_metrics["eval/episode_reward"]
    best_policy_state = _save_policy_state(policy_module)

    for epoch in range(cfg.pilco.epochs):
        with timeit("gp_fit"):
            base_world_model = GPWorldModel(
                obs_dim=obs_dim, action_dim=action_dim
            ).to(env.device)
            base_world_model.fit(rollout)
            base_world_model.freeze_and_detach()

        torchrl_logger.info(
            f"  GP ls: {base_world_model.lengthscales.tolist()}\n"
            f"  GP var: {base_world_model.variances.squeeze().tolist()}\n"
            f"  GP noise: {base_world_model.noises.tolist()}",
        )

        world_model_module = TensorDictModule(
            module=base_world_model,
            in_keys=["action", "observation"],
            out_keys=[("next_observation", "mean"), ("next_observation", "var")],
        )
        imagined_env = ImaginedEnv(
            world_model_module=world_model_module, base_env=env
        )
        obs_spec = Composite(
            observation=Composite(
                mean=Unbounded(shape=(obs_dim,), dtype=dtype),
                var=Unbounded(shape=(obs_dim, obs_dim), dtype=dtype),
            ),
        )
        imagined_env.observation_spec = obs_spec.expand(
            imagined_env.batch_size
        ).clone()
        imagined_env.state_spec = obs_spec.expand(imagined_env.batch_size).clone()
        reset_imagination = initial_observation.expand(*imagined_env.batch_size)

        pre_optim_state = _save_policy_state(policy_module)

        def _optimize_policy():
            """Run one LBFGS optimization and return final loss."""
            optimizer = torch.optim.LBFGS(
                policy_module.parameters(),
                max_iter=cfg.pilco.policy_training_steps,
                line_search_fn="strong_wolfe",
            )
            final_loss = [float("inf")]

            def closure():
                optimizer.zero_grad()
                imagined_data = imagined_env.rollout(
                    max_steps=cfg.pilco.horizon,
                    policy=policy_module,
                    tensordict=reset_imagination,
                )
                loss_td = cost_module(imagined_data)
                loss = loss_td.get("loss_cost").sum(dim=-1).mean()
                loss.backward()
                final_loss[0] = loss.item()
                return loss

            optimizer.step(closure)
            return final_loss[0]

        n_restarts = cfg.pilco.get("policy_restarts", 1)

        with timeit("policy_optim"):
            best_imag_loss = _optimize_policy()
            best_optim_state = _save_policy_state(policy_module)

            for _ in range(n_restarts - 1):
                # Perturb from the pre-optimization state
                _restore_policy_state(policy_module, pre_optim_state)
                for p in policy_module.parameters():
                    p.data.add_(torch.randn_like(p) * 0.3)
                candidate_loss = _optimize_policy()
                if candidate_loss < best_imag_loss:
                    best_imag_loss = candidate_loss
                    best_optim_state = _save_policy_state(policy_module)

            _restore_policy_state(policy_module, best_optim_state)

        last_loss = [best_imag_loss]
        step_counter = [0]

        with timeit("eval"), torch.no_grad():
            test_rollout, eval_metrics = eval_policy(
                eval_env, policy_module, reset_td, eval_max_steps
            )
            eval_cost = cost_module(test_rollout).get("loss_cost").sum().item()
            eval_metrics["eval/pilco_cost"] = eval_cost

        eval_reward = eval_metrics["eval/episode_reward"]

        # Revert policy if real performance degraded significantly
        if eval_reward < best_reward - 100:
            torchrl_logger.info(
                f"  Reverting policy: {eval_reward:.2f} < {best_reward:.2f} - 100"
            )
            _restore_policy_state(policy_module, best_policy_state)
            eval_metrics["train/reverted"] = 1.0
        else:
            eval_metrics["train/reverted"] = 0.0
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_policy_state = _save_policy_state(policy_module)

        logger_step = (epoch + 1) * cfg.pilco.policy_training_steps
        metrics = {
            **eval_metrics,
            "train/trajectory_cost": last_loss[0],
            "train/lbfgs_evals": step_counter[0],
            "train/dataset_size": len(rollout),
            "train/best_reward": best_reward,
        }
        metrics.update(timeit.todict(prefix="time"))
        if logger:
            logger.log_metrics(metrics, logger_step)

        torchrl_logger.info(
            f"Epoch {epoch}/{cfg.pilco.epochs} | "
            f"reward: {eval_reward:.2f} | "
            f"pilco cost: {eval_cost:.2f} | "
            f"imag cost: {last_loss[0]:.4f} | "
            f"best: {best_reward:.2f} | "
            f"dataset: {len(rollout)}",
        )

        # Always keep initial exploration data; only trim policy-collected data
        new_data = flatten_eval_rollout(test_rollout)
        policy_data = rollout[n_initial:]
        policy_data = tensordict.cat([policy_data, new_data], dim=0)
        max_policy_data = cfg.pilco.max_rollout_length - n_initial
        if len(policy_data) > max_policy_data:
            policy_data = policy_data[-max_policy_data:]
        rollout = tensordict.cat([initial_rollout, policy_data], dim=0)

    return policy_module


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device) if cfg.device else get_available_device()

    env = make_env(device, max_steps=cfg.pilco.eval_max_steps)

    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PILCO", "Pendulum")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="pilco",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    pilco_loop(cfg, env, logger=logger)

    if not env.is_closed:
        env.close()


if __name__ == "__main__":
    main()
