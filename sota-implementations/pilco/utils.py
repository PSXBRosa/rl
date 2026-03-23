import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs import TransformedEnv
from torchrl.envs.custom.pendulum import PendulumEnv
from torchrl.envs.transforms import (
    DTypeCastTransform,
    RewardSum,
    StepCounter,
    Transform,
)


class PendulumObservationTransform(Transform):
    """Encodes PendulumEnv's (th, thdot) as observation = [cos(th), sin(th), thdot]."""

    def __init__(self):
        super().__init__(in_keys=["th", "thdot"], out_keys=["observation"])

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        th = td.get("th")
        thdot = td.get("thdot")
        td.set("observation", torch.stack([th.cos(), th.sin(), thdot], dim=-1))
        return td

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        th_spec = observation_spec["th"]
        observation_spec["observation"] = Unbounded(
            shape=(3,), dtype=th_spec.dtype, device=th_spec.device
        )
        return observation_spec


def make_env(device: str | torch.device, max_steps: int = 200) -> TransformedEnv:
    """Creates a PendulumEnv with PILCO-compatible observation encoding."""
    env = TransformedEnv(PendulumEnv(device=device))
    env.append_transform(PendulumObservationTransform())
    env.append_transform(DTypeCastTransform(torch.float32, torch.float64))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=max_steps))
    return env
