from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import math
from itertools import chain
from typing import Optional,Any
from torch.amp import GradScaler

from src.rl.flashsac.flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler

class FlashSAC:
    """The actor critic module."""

    def __init__(
        self,
        policy,
        device,
        buffer_max_length= 1_000_000,
        buffer_min_length=10_000,
        sample_batch_size=2048,
        asymmetric_observation=False,
        normalize_reward=True,
        num_action=29,
        gamma=0.99,
        learning_rate_peak=3e-4,
        temp_target_sigma=0.15,
        use_amp=False,
        n_step=1,
        actor_update_period=2,
        learning_rate_init=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1000,
        learning_rate_decay_step=100000,
        num_interaction_steps=10000,
        updates_per_interaction_step=2,
        learning_rate_decay_rate=1,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        # device-related parameters
        self.device = device
        self.use_amp=use_amp
        self.gamma=gamma
        self.n_step=n_step
        self.actor_update_period=actor_update_period
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        self.learning_rate_warmup_step=learning_rate_warmup_rate*num_interaction_steps*updates_per_interaction_step
        self.learning_rate_decay_step=learning_rate_decay_rate*num_interaction_steps*updates_per_interaction_step
        warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
            init_value=learning_rate_init,
            peak_value=learning_rate_peak,
            end_value=learning_rate_end,
            warmup_steps=self.learning_rate_warmup_step,
            decay_steps=self.learning_rate_decay_step,
        )
        self.learning_rate=warmup_cosine_decay_lr
        self.asymmetric_observation=asymmetric_observation
        self.normalize_reward=normalize_reward
        # FlashSAC components
        self.policy = policy
        self.policy.to(self.device)
        use_fused = device == "cuda" and torch.cuda.is_available()
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=learning_rate_peak, fused=use_fused)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda step: warmup_cosine_decay_lr(step) / learning_rate_peak,
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=learning_rate_peak,
            fused=use_fused,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer,
            lr_lambda=lambda step: warmup_cosine_decay_lr(step) / learning_rate_peak,
        )
        self.policy.target_critic.load_state_dict(self.policy.critic.state_dict())
        self.temp_optimizer = optim.Adam(
            self.policy.temperature.parameters(),
            lr=learning_rate_peak/100.0, # lei:
            fused=use_fused,
        )
        self.temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.temp_optimizer,
            lr_lambda=lambda step: warmup_cosine_decay_lr(step) / learning_rate_peak,
        )

        self.temp_target_entropy = 0.5 *num_action * math.log(2 * math.pi * math.e *temp_target_sigma**2)
        self.policy.actor.normalize_parameters() 
        self.policy.critic.normalize_parameters()
        self.policy.target_critic.normalize_parameters()
        #self.policy.setup_weight_normalization()

    def act(self, prev_transition,training):
        temperature = 1.0 if training else 0.0
        observations = prev_transition["next_observation"]["policy"]
        if self.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        observations = torch.as_tensor(observations, dtype=torch.float32)
        action=self.policy.act(observations,temperature,training)
        return action

    def update_actor(
        self,
        batch,
        bc_alpha,
        device: torch.device,
        use_amp: bool,
        grad_scaler: Optional[GradScaler],
    ) -> dict[str, torch.Tensor]:
        torch.autograd.set_detect_anomaly(True)
        """Update actor network.

        Args:
            actor: Actor network.
            critic: Critic network.
            temperature: Temperature network.
            batch: Batch of transitions.
            bc_alpha: BC regularization coefficient.
            device: Device to use.
            use_amp: Whether to use automatic mixed precision.
            grad_scaler: GradScaler for FP16 AMP.
        """
        if isinstance(device, str):
            device = torch.device(device)

        #with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        actor_obs_all = torch.cat([
            batch["actor_observation"], 
            batch["actor_next_observation"]
        ], dim=0)
        actor_obs_all = actor_obs_all.reshape(actor_obs_all.size(0), -1)
        actions_all, info = self.policy.actor(
            observations=actor_obs_all,
            training=True,
        )
        log_probs_all = info["log_prob"]

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # Disable critic gradients to prevent CUDA graph overwriting
        self.policy.critic.requires_grad_(False)
        critic_obs=batch["observation"]["critic"]
        critic_obs = critic_obs.reshape(critic_obs.size(0), -1) 
        qs, q_infos = self.policy.evaluate(
            observations=critic_obs,
            actions=actions,
            training=False,
        )
        q = torch.minimum(qs[0], qs[1])
        self.policy.critic.requires_grad_(True)

        temp_value = self.policy.temperature().detach() # lei:
        # temp_value = 0.001
        actor_loss = (log_probs * temp_value - q).mean()

        if bc_alpha > 0:
            # https://arxiv.org/abs/2306.02451
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        entropy = (-log_probs).mean()
        mean_action = actions.mean()

        #############
        # Gradient step
        assert self.actor_optimizer is not None
        self.actor_optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert grad_scaler is not None
            grad_scaler.scale(actor_loss).backward()
            grad_scaler.step(self.actor_optimizer)
            grad_scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()

        # LR scheduler
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        # Weight Normalization
        # NOTE: Make sure you finish all computation before this (e.g., computing info values)
        self.policy.actor.normalize_parameters()

        update_info = {
            "loss": actor_loss,
            "entropy": entropy,
            "log_prob": log_probs.mean(),
            "mean_action": mean_action,
        }
        update_info = add_prefix_to_keys(update_info, "actor")

        return update_info

    def update_critic(
        self,
        batch,
        min_v: float,
        max_v: float,
        num_bins: int,
        gamma: float,
        n_step: int,
        device: torch.device,
        use_amp: bool,
        grad_scaler: Optional[GradScaler],
    ) -> dict[str, torch.Tensor]:
        """Update critic network.

        Args:  
            actor: Actor network.
            critic: Critic network.
            target_critic: Target critic network.
            temperature: Temperature network.
            batch: Batch of transitions.
            min_v: Minimum value for categorical distribution.
            max_v: Maximum value for categorical distribution.
            num_bins: Number of bins for categorical distribution.
            gamma: Discount factor.
            n_step: N-step return.
            device: Device to use.
            use_amp: Whether to use automatic mixed precision.
            grad_scaler: GradScaler for FP16 AMP.
        """
        if isinstance(device, str):
            device = torch.device(device)
        # with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
        #############
        # Compute target values
        actor_next_obs =batch["actor_next_observation"]
        actor_next_obs=actor_next_obs.view(actor_next_obs.size(0), -1)
        with torch.no_grad():
            next_actions, info = self.policy.actor(
                observations=actor_next_obs,
                training=True,
            )
            # Clone variables to prevent overwriting
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()
            temp_value = self.policy.temperature()
            next_actor_entropy = temp_value * next_actor_log_probs
            
            batch["critic_observation"] = batch["observation"]["critic"]
            batch["critic_next_observation"] = batch["next_observation"]["critic"]
            obs_all = torch.cat([batch["critic_observation"].clone(), batch["critic_next_observation"].clone()], dim=0)  # type: ignore
            obs_all = obs_all.reshape(obs_all.size(0), -1) 
            act_all = torch.cat([batch["action"].clone(), next_actions], dim=0)  # type: ignore
            qs_all, q_infos_all = self.policy.target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
            )
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],  # type: ignore
                done=batch["done"],  # type: ignore
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()
        # Compute predicted q-value
        pred_qs_all, pred_q_infos = self.policy.critic(
            observations=obs_all,
            actions=act_all,
            training=False,
        )
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"].clone(), 2, dim=1)[0]
        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

        ##########
        # Gradient step
        assert self.critic_optimizer is not None
        self.critic_optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert grad_scaler is not None
            grad_scaler.scale(critic_loss).backward()
            grad_scaler.step(self.critic_optimizer)
            grad_scaler.update()
        else:
            critic_loss.backward()  # type: ignore
            self.critic_optimizer.step()

        # LR scheduler
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # Weight Normalization
        self.policy.critic.normalize_parameters()

        update_info = {
            "loss": critic_loss,
            "max_entropy_bonus": max_entropy_bonus,
        }
        update_info = add_prefix_to_keys(update_info, "critic")
        return update_info

    @torch.no_grad()
    def update_target_network(
        self,
    ) -> dict[str, torch.Tensor]:
        # Use prepared/compiled EMA function for update
        self.policy.soft_update_target_critic()
        info: dict[str, torch.Tensor] = {}
        return info

    def update_temperature(
        self,
        entropy: torch.Tensor,
        log_prob:torch.Tensor,
        target_entropy :torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update temperature network.

        Args:
            temperature: Temperature network.
            entropy: Current entropy value.
            target_entropy: Target entropy value.
        """
        self.temp_target_entropy=target_entropy
        temperature_value = self.policy.temperature().clone()
        temperature_loss = temperature_value * (entropy.detach() - self.temp_target_entropy ).mean()
        log_alpha = self.policy.temperature.log_temp
        alpha = log_alpha.exp()
        assert self.temp_optimizer is not None
        self.temp_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temp_optimizer.step()
        if self.temp_scheduler is not None:
            self.temp_scheduler.step()

        update_info = {
            "value": temperature_value,
            "loss": temperature_loss,
        }
        update_info = add_prefix_to_keys(update_info, "temperature")

        return update_info
    
    def _recursive_to_device(self, data, device):
        if isinstance(data, dict):
            return {k: self._recursive_to_device(v, device) for k, v in data.items()}
        else:
            return data.to(device, non_blocking=True)
        
    def update(self,batch,do_actor_update,grad_scaler):  # noqa: C901
        for k, v in batch.items():
            batch[k] = self._recursive_to_device(v, self.device)

        if self.asymmetric_observation:
            _actor_observation_dim=batch["observation"]["policy"].size(-1)
            batch["actor_observation"] = {
                "policy": batch["observation"]["policy"], 
                "critic": batch["observation"]["critic"][:,:_actor_observation_dim]
            }
            batch["actor_next_observation"] = {
                "policy": batch["next_observation"]["policy"], 
                "critic": batch["next_observation"]["critic"][:,:_actor_observation_dim]
            }            
        else:
            batch["actor_observation"] = batch["observation"]["policy"]
            batch["actor_next_observation"] = batch["next_observation"]["policy"]
        
        if do_actor_update:
            actor_info = self.update_actor(
                batch=batch,  # type: ignore
                bc_alpha=self.policy.actor_bc_alpha,
                device=self.device,
                use_amp=self.use_amp,
                grad_scaler=grad_scaler,
            )
            temperature_info = self.update_temperature(
                entropy=actor_info["actor/entropy"],
                log_prob=actor_info["actor/log_prob"],
                target_entropy=self.temp_target_entropy,
            )
        else:
            actor_info = {}
            temperature_info = {}          
        critic_info = self.update_critic(
            batch=batch,  # type: ignore
            min_v=self.policy.critic_min_v,
            max_v=self.policy.critic_max_v,
            num_bins=self.policy.critic_num_bins,
            gamma=self.gamma,
            n_step=self.n_step,
            device=self.device,
            use_amp=self.use_amp,
            grad_scaler=grad_scaler,
        )
        target_critic_info = self.update_target_network()
        # Merge all info dicts
        update_info = {
            **actor_info,
            **critic_info,
            **target_critic_info,
            **temperature_info,
        }
        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()

    

        return update_info

        """
        Helper functions
        """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def construct_algorithm(obs, env, cfg, device):
        return None
                


def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,  
    next_q_log_probs: torch.Tensor, 
) -> torch.Tensor:
    """Select log-probs from the min-Q critic and return the next-obs half (batch, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[
        0
    ] 
    return selected

@torch.compile
def _compute_categorical_td_target(
    target_log_probs: torch.Tensor, 
    reward: torch.Tensor,  
    done: torch.Tensor,
    actor_entropy: torch.Tensor,  
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # target_bin_values
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # update indices
    b = (target_bin_values - min_v) / bin_width
    b = torch.clamp(b, 0, num_bins - 1)
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    
    frac = b - lower.float()

    # Compute target probabilities using exp
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    # Allocate output tensor
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
    # Scatter operations
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)
    return target_probs
