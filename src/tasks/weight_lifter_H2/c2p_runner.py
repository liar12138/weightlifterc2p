"""Online supervised C2P runner driven by a frozen privileged teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from src.tasks.weight_lifter_H2.networks.barbell_c2p import BarbellC2PModel
from src.tasks.weight_lifter_H2.networks.priv_res import PrivResMLP


class OnlineC2PRunner:
    """Collect simulation pairs with the teacher and update only C2P."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ) -> None:
        del registry_name
        self.env = env
        self.cfg = train_cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir or "logs/c2p")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_learning_iteration = 0

        obs = self.env.get_observations()
        actor = obs["actor"]
        depth = obs["depth_camera"]
        target = obs["c2p_target"]
        if actor.ndim != 2 or target.shape[-1] != 16:
            raise ValueError(
                f"C2P groups have wrong shapes: actor={tuple(actor.shape)}, "
                f"target={tuple(target.shape)}"
            )
        self.model = BarbellC2PModel(actor.shape[-1], tuple(depth.shape[1:])).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("c2p_learning_rate", 1.0e-4)),
            weight_decay=float(train_cfg.get("c2p_weight_decay", 1.0e-5)),
        )
        self.c2p_epochs = int(train_cfg.get("c2p_epochs", 5))
        self.c2p_mini_batches = int(train_cfg.get("c2p_mini_batches", 4))
        self.max_grad_norm = float(train_cfg.get("c2p_max_grad_norm", 1.0))
        self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
        self.save_interval = int(train_cfg.get("save_interval", 100))

        teacher_checkpoint = str(
            train_cfg.get("teacher_checkpoint", "src/assets/model_28300.pt")
        )
        self.teacher = PrivResMLP(
            obs=obs,
            obs_groups={"actor": ["teacher_actor"], "critic": ["teacher_actor"]},
            obs_set="actor",
            output_dim=int(self.env.unwrapped.action_manager.total_action_dim),
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "scalar",
            },
        ).to(self.device)
        checkpoint = torch.load(teacher_checkpoint, map_location=self.device, weights_only=False)
        self.teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        print(
            f"[C2P] actor={tuple(actor.shape)} depth={tuple(depth.shape)} "
            f"target={tuple(target.shape)} teacher={teacher_checkpoint}"
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        del repo_file_path

    def _teacher_obs(self, obs: TensorDict) -> TensorDict:
        return TensorDict(
            {"teacher_actor": obs["teacher_actor"], "barbell": obs["barbell"]},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def _update(self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        proprio = torch.cat([item[0] for item in samples], dim=0).to(self.device)
        depth = torch.cat([item[1] for item in samples], dim=0).to(self.device)
        target = torch.cat([item[2] for item in samples], dim=0).to(self.device)
        count = proprio.shape[0]
        batch_size = max(1, (count + self.c2p_mini_batches - 1) // self.c2p_mini_batches)
        total_loss = 0.0
        updates = 0
        self.model.train()
        for _ in range(self.c2p_epochs):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, batch_size):
                idx = order[start : start + batch_size]
                loss = self.model.loss(proprio[idx], depth[idx], target[idx])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
        self.model.eval()
        return total_loss / max(1, updates)

    def save(self, path: str, infos: dict | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos or {},
                "config": self.cfg,
            },
            path,
        )

    def load(self, path: str, *args: Any, **kwargs: Any) -> dict:
        del args, kwargs
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        return checkpoint.get("infos", {})

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del init_at_random_ep_len
        obs = self.env.get_observations().to(self.device)
        for iteration in range(
            self.current_learning_iteration,
            self.current_learning_iteration + int(num_learning_iterations),
        ):
            samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.teacher(self._teacher_obs(obs))
                    samples.append(
                        (
                            obs["actor"].detach().cpu(),
                            obs["depth_camera"].detach().cpu(),
                            obs["c2p_target"].detach().cpu(),
                        )
                    )
                    obs, _, _, _ = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
            loss = self._update(samples)
            self.current_learning_iteration = iteration + 1
            print(f"[C2P] iteration={self.current_learning_iteration} loss={loss:.6f}")
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(str(self.log_dir / f"c2p_model_{self.current_learning_iteration}.pt"))
        self.save(str(self.log_dir / f"c2p_model_{self.current_learning_iteration}.pt"))

    def get_inference_policy(self, device: str | None = None) -> BarbellC2PModel:
        self.model.eval()
        return self.model.to(device or self.device)
