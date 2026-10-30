"""C2P environment and runner configuration."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from ..c2p_observation import c2p_target_16d
from ..observation import (
    barbell_qpos,
    bellbar_vector,
    left_hand_plate_vector,
    right_hand_plate_vector,
)
from .env_cfgs import unitree_h2_pull_up_barbell_env_cfg
from src.sensors.h2.d435c import deploy_depth_camera_to


def unitree_h2_pull_up_barbell_c2p_env_cfg(play: bool = False):
    history_length = int(os.environ.get("C2P_HISTORY_LENGTH", "10"))
    if history_length < 1:
        raise ValueError("C2P_HISTORY_LENGTH must be positive")
    cfg = unitree_h2_pull_up_barbell_env_cfg(play=play)
    cfg.commands["motion"].motion_file = "src/assets/motion/h2o6/1_h2o6_new_s6.npz"
    cfg = deploy_depth_camera_to(cfg, depth_history_length=history_length)

    # The teacher receives exactly the privileged task's original 274-D actor input.
    base_terms = copy.deepcopy(cfg.observations["actor"].terms)
    cfg.observations["teacher_actor"] = ObservationGroupCfg(
        terms=base_terms, concatenate_terms=True, enable_corruption=False
    )
    cfg.observations["barbell"] = ObservationGroupCfg(
        terms={
            "barbell_qpos": ObservationTermCfg(func=barbell_qpos),
            "bellbar_vector": ObservationTermCfg(
                func=bellbar_vector, params={"command_name": "motion"}
            ),
            "left_hand_plate_vector": ObservationTermCfg(
                func=left_hand_plate_vector, params={"command_name": "motion"}
            ),
            "right_hand_plate_vector": ObservationTermCfg(
                func=right_hand_plate_vector, params={"command_name": "motion"}
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    cfg.observations["c2p_target"] = ObservationGroupCfg(
        terms={
            "state_16d": ObservationTermCfg(
                func=c2p_target_16d, params={"command_name": "motion"}
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    if play:
        cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    return cfg


@dataclass
class C2PRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Compatibility wrapper so scripts/train.py can launch OnlineC2PRunner."""

    c2p_learning_rate: float = 1.0e-4
    c2p_weight_decay: float = 1.0e-5
    c2p_epochs: int = 5
    c2p_mini_batches: int = 4
    c2p_max_grad_norm: float = 1.0
    teacher_checkpoint: str = "src/assets/model_28300.pt"


def unitree_h2_c2p_runner_cfg() -> C2PRunnerCfg:
    return C2PRunnerCfg(
        actor=RslRlModelCfg(class_name="rsl_rl.models.mlp_model:MLPModel"),
        critic=RslRlModelCfg(class_name="rsl_rl.models.mlp_model:MLPModel"),
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        experiment_name="h2_barbell_c2p",
        run_name="c2p",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
    )
