import tqdm,torch,os,time
import numpy as np
from collections import deque
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.rl.flashsac.flash_rl.agents.flashSAC.agent import (
    FlashSACAgent,
    FlashSACConfig,
)
from src.rl.flashsac.flash_rl.agents.flashSAC.network import FlashSACActor
from typing import Any, Dict
from dataclasses import fields

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}min"
    else:
        return f"{seconds / 3600:.2f}h"

def create_agent(
    observation_space ,
    num_actions ,
    actor_observation_size,
    critic_observation_size,
    cfg_dict,
) :
    cfg_dict = {str(k): v for k, v in cfg_dict.items()}
 

    valid_keys = {f.name for f in fields(FlashSACConfig)}

    filtered_cfg = {
        k: v
        for k, v in cfg_dict.items()
        if k in valid_keys
    }
    config = FlashSACConfig(**filtered_cfg)  # type: ignore
    agent = FlashSACAgent(observation_space.cpu().numpy(), torch.zeros([num_actions],dtype=torch.float32).numpy(), actor_observation_size,actor_observation_size,critic_observation_size, config)

    return agent

class FlashSACRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir = None, device = "cpu"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        self.log_dir=log_dir
        self.actor=None
        super().__init__(env, train_cfg, log_dir, device)
        self.max_iterations = self.cfg.get("max_iterations",2000000)

    def _construct_algorithm(self,obs):
        return None
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:

        observations, env_infos = self.env.reset()
        self.logger.init_logging_writer()
        #############################
        # agent
        #############################

        # Since the network architecture is typically tied to the learning algorithm,
        #   we opted not to fully modularize the network for the sake of readability.
        # Therefore, for each algorithm, the network is implemented within its respective directory.
        self.alg_cfg = self.cfg["algorithm"]
        self.policy_cfg = self.cfg["policy"]
        agent = create_agent(
            observation_space=torch.cat([observations['actor'],observations['critic']],dim = -1),
            num_actions = self.env.num_actions,
            actor_observation_size = observations["actor"].size(-1),
            critic_observation_size = observations["critic"].size(-1),
            cfg_dict=self.alg_cfg | self.policy_cfg | self.cfg
        )
        print("agent",agent)

        #############################
        # train
        #############################

        # start training

        transition = None
        update_counter = 0
        total_rewards = 0
        episode_buffer = []
        update_info = {}
        episode_lengths = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        recent_episode_lengths = deque(maxlen=200)
        start_time = time.time()

        for interaction_step in range(1, int(self.max_iterations + 1)):
            # collect data. use random actions until agent.can_start_training()
            if agent.can_start_training() and transition is not None:
                observations = transition['next_observation']
                actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
                actions=torch.tensor(actions,dtype=torch.float32)
            else:
                observations = self.env.get_observations()
                observations = torch.cat([observations['actor'],observations['critic']],dim = -1)
                single_low = -0.3
                single_high = 0.3
                low = torch.tensor(single_low, device=self.env.device, dtype=torch.float32)
                high = torch.tensor(single_high, device=self.env.device, dtype=torch.float32)
                random_tensor = torch.rand(self.env.num_envs, self.env.num_actions, device=self.env.device)
                actions = low + random_tensor * (high - low)

            # Step the environment

            next_observation, rewards, terminateds, extras = self.env.step(actions.to(self.env.device))
            next_observation = torch.cat([next_observation['actor'],next_observation['critic']],dim = -1)
            next_observation = next_observation.clone()
            for env_idx in range(terminateds.size(0)):
                if terminateds[env_idx]:
                    next_observation[env_idx] = torch.cat([extras["final_obs"]['actor'],extras["final_obs"]['critic']],dim = -1)[env_idx]

            transition = {
                "observation": observations,
                "action": actions,
                "reward": rewards,
                "terminated": terminateds,
                "next_observation": next_observation,
            }
            agent.process_transition(transition)

            ############# Returns ##################
            total_rewards += rewards.detach().cpu()
            done_indices = np.where(terminateds.detach().cpu().numpy())[0]

            for i in done_indices:
                episode_buffer.append(total_rewards[i].item())
                total_rewards[i] = 0.0   # reset

            episode_lengths += 1

            done_env_ids = torch.nonzero(terminateds, as_tuple=False).squeeze(-1)

            if len(done_env_ids) > 0:
                recent_episode_lengths.extend(
                    episode_lengths[done_env_ids].cpu().tolist()
                )

                episode_lengths[done_env_ids] = 0
            
            elapsed_time = time.time() - start_time
            ###############################

            if agent.can_start_training():
                # update network
                update_counter += self.alg_cfg.get("updates_per_interaction_step",2)
                while update_counter >= 1:
                    update_info = update_info | agent.update() | extras["log"]
                    update_counter -= 1

                if interaction_step % self.cfg.get("log_interval",200) == 0:
                    log_update_info(update_info | agent.get_metrics() | {
                        "returns":np.mean(episode_buffer),
                        "episode_length": sum(recent_episode_lengths) / len(recent_episode_lengths)
                    },step=interaction_step,elapsed_time = elapsed_time,ETA = (self.max_iterations - interaction_step) * elapsed_time / interaction_step)
                    
                if interaction_step % self.cfg.get("save_interval",2000) == 0: 
                    self.save(os.path.join(self.log_dir, f"model_{interaction_step}.pt"),agent=agent)

    def save(self, path, agent, infos=None):
        actor=agent._actor.network_ori
        self.actor=actor
         # -- Save model
        saved_dict = {
            # actor 
            "model_state_dict": agent._actor.network.state_dict(),
            "optimizer_state_dict": agent._actor.optimizer.state_dict(),
            "scheduler_state_dict": agent._actor.scheduler.state_dict() ,
            # critic
            "critic_state_dict": agent._critic.network.state_dict(),
            "critic_optimizer_state_dict": agent._critic.optimizer.state_dict(),
            "critic_scheduler_state_dict": agent._critic.scheduler.state_dict() ,
            # target_critic
            "target_critic_state_dict": agent._target_critic.network.state_dict(),
            # temperature
            "temperature_state_dict": agent._temperature.network.state_dict(),
            "temperature_optimizer_state_dict": agent._temperature.optimizer.state_dict(),
            "temperature_scheduler_state_dict": agent._temperature.scheduler.state_dict() ,            
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        print("model save to:",path) 

    def act_inference(self,obs):
        act,_ = self.loaded_actor(obs['actor'],training=False)
        return act
    
    def get_inference_policy(self, device=None):
        self.loaded_actor.eval()
        return self.act_inference

    def load(self, path, load_optimizer = True, map_location = None):
        obs = self.env.get_observations()
        self.loaded_actor =  FlashSACActor(
            num_blocks=self.cfg["policy"].get("actor_num_blocks"),
            input_dim=obs['actor'].shape[-1],
            hidden_dim=self.cfg["policy"].get("actor_hidden_dim"),
            action_dim=self.env.num_actions,
        ).to(self.env.device)

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        actor_state_dict = checkpoint["model_state_dict"]

        # Compatible with checkpoints saved by torch.compile.
        actor_state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in actor_state_dict.items()
        }

        self.loaded_actor.load_state_dict(
            actor_state_dict,
            strict=True,
        )
 

def _to_scalar(v):
    """Convert tensor to python scalar."""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    return v


def _fmt(v):
    """Pretty format value."""
    v = _to_scalar(v)

    if isinstance(v, float):
        return f"{v:10.4f}"
    if isinstance(v, int):
        return f"{v:10d}"
    return str(v)


def log_update_info(
    update_info: Dict[str, Any],
    step: int | None = None,
    elapsed_time: float | None = None,
    ETA: float | None = None,
):
    sections = {
        "Actor": [
            "actor/loss",
            "actor/entropy",
            "actor/mean_action",
        ],
        "Critic": [
            "critic/loss",
            "critic/max_entropy_bonus",
        ],
        "Temperature": [
            "temperature/value",
            "temperature/loss",
        ],
        "Episode Reward": [
            "Episode_Reward/track_linear_velocity",
            "Episode_Reward/track_angular_velocity",
            "Episode_Reward/flat_orientation_l2",
            "Episode_Reward/pose",
            "Episode_Reward/body_ang_vel",
            "Episode_Reward/angular_momentum",
            "Episode_Reward/is_terminated",
            "Episode_Reward/joint_acc_l2",
            "Episode_Reward/joint_pos_limits",
            "Episode_Reward/action_rate_l2",
            "Episode_Reward/foot_air_time",
            "Episode_Reward/foot_clearance",
            "Episode_Reward/foot_slip",
            "Episode_Reward/soft_landing",
            "Episode_Reward/self_collisions",
        ],
        "Metrics": [
            "Metrics/twist/error_vel_xy",
            "Metrics/twist/error_vel_yaw",
            "Metrics/angular_momentum_mean",
            "Metrics/slip_velocity_mean",
            "Metrics/landing_force_mean",
        ],
        "Termination": [
            "Episode_Termination/time_out",
            "Episode_Termination/fell_over",
        ],
    }

    header = []

    if step is not None:
        header.append(f"step={step:,}")

    if elapsed_time is not None:
        header.append(f"time={format_time(elapsed_time)}")
        header.append(f"ETA={format_time(ETA)}")

    header_str = " | ".join(header)

    print("\n" + "=" * 100)
    print(f" Update Info {header_str}")
    print("=" * 100)

    used_keys = set()

    for section_name, keys in sections.items():
        present_keys = [k for k in keys if k in update_info]

        if not present_keys:
            continue

        print(f"\n[{section_name}]")

        for k in present_keys:
            used_keys.add(k)

            short_name = k.split("/")[-1]
            print(f"  {short_name:<30} {_fmt(update_info[k])}")

    extra_keys = sorted(set(update_info) - used_keys)

    if extra_keys:
        print("\n[Extra]")
        for k in extra_keys:
            print(f"  {k:<40} {_fmt(update_info[k])}")

    print("=" * 100, flush=True)