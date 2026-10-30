"""Extended task registry with PPO + FastSAC configurations."""

from __future__ import annotations

from copy import deepcopy

from mjlab.tasks import registry as mjlab_registry

from src.rl.fastsac.config import HolosomaFastSACRunnerCfg

_REGISTRY_EXT: dict[str, HolosomaFastSACRunnerCfg] = {}
_RUNNER_EXT: dict[str, type | None] = {}


def extend_mjlab_task_with_fastsac(
  task_id: str,
  rl_cfg_fastsac: HolosomaFastSACRunnerCfg,
  runner_cls_fastsac: type | None = None,
) -> None:
  if task_id not in mjlab_registry.list_tasks():
    raise ValueError(f"Task {task_id!r} is not registered in mjlab registry")
  if task_id in _REGISTRY_EXT:
    raise ValueError(f"Task {task_id!r} already has FastSAC config")
  _REGISTRY_EXT[task_id] = rl_cfg_fastsac
  _RUNNER_EXT[task_id] = runner_cls_fastsac


def fastsac_task_ids() -> tuple[str, ...]:
  return tuple(sorted(_REGISTRY_EXT))


def assert_fastsac_task(task_id: str) -> None:
  if task_id in _REGISTRY_EXT:
    return
  allowed = ", ".join(fastsac_task_ids()) or "(none registered)"
  raise ValueError(f"FastSAC is only supported for: {allowed}. Got {task_id!r}.")


def load_fastsac_rl_cfg(task_name: str) -> HolosomaFastSACRunnerCfg:
  assert_fastsac_task(task_name)
  return deepcopy(_REGISTRY_EXT[task_name])


def load_fastsac_runner_cls(task_name: str) -> type | None:
  assert_fastsac_task(task_name)
  return _RUNNER_EXT.get(task_name)
