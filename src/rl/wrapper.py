import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.spaces import Box, batch_space
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

class BodyOnlyRslRlVecEnvWrapper(RslRlVecEnvWrapper):


    def __init__(self,env: ManagerBasedRlEnv, clip_actions: float | None = None):
        super().__init__(
            env=env,
            clip_actions=clip_actions,
        )

        self.num_actions = 31

        # ---------------------------------------------------------
        # 53维action中身体的index
        # 24 + 7 = 31
        # ---------------------------------------------------------
        self.body_ids = torch.tensor(
            [
                # 左腿
                0, 1, 2, 3, 4, 5,

                # 右腿
                6, 7, 8, 9, 10, 11,

                # 腰
                12, 13, 14,

                # 头
                15, 16,

                # 左臂 + 左手腕
                17, 18, 19, 20, 21, 22, 23,

                # 右臂 + 右手腕
                35, 36, 37, 38, 39, 40, 41,
            ],
            dtype=torch.long,
            device=self.device,
        )

        # ---------------------------------------------------------
        # 53维action中手指的index
        # 左11 + 右11 = 22
        # ---------------------------------------------------------
        self.hand_ids = torch.tensor(
            [
                # 左手
                24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34,

                # 右手
                42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52,
            ],
            dtype=torch.long,
            device=self.device,
        )

        # ---------------------------------------------------------
        # 父类已经按照53维设置过action space。
        # 这里重新改成31维，让rsl_rl / PPO看到31维。
        # ---------------------------------------------------------
        self._modify_body_action_space()

        # ---------------------------------------------------------
        # 保存当前policy对应的observation。
        # 后面step时，从这里读取motion reference joint_pos。
        # ---------------------------------------------------------
        self._last_obs = self.get_observations()

    def _modify_body_action_space(self) -> None:
        """
        修改rsl_rl看到的action space为31维。
        底层ManagerBasedRlEnv内部仍然是53维。
        """

        if self.clip_actions is None:
            low = -float("inf")
            high = float("inf")
        else:
            low = -self.clip_actions
            high = self.clip_actions

        self.unwrapped.single_action_space = Box(
            shape=(self.num_actions,),
            low=low,
            high=high,
        )

        self.unwrapped.action_space = batch_space(
            self.unwrapped.single_action_space,
            self.num_envs,
        )

    def get_observations(self) -> TensorDict:
        """
        获取observation，同时保存下来。

        你已经确认：
            obs["actor"][:, :53]
        就是motion reference joint_pos。
        """

        obs = super().get_observations()

        self._last_obs = obs

        return obs

    def reset(self) -> tuple[TensorDict, dict]:
        """
        Reset环境，同时保存reset后的obs。
        """

        obs, extras = super().reset()

        self._last_obs = obs

        return obs, extras

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:

        if actions.ndim != 2 or actions.shape[1] != self.num_actions:
            raise ValueError(
                f"BodyOnlyRslRlVecEnvWrapper expected actions "
                f"with shape [num_envs, {self.num_actions}], "
                f"but got {tuple(actions.shape)}."
            )

        if self.clip_actions is not None:
            actions = torch.clamp(
                actions,
                -self.clip_actions,
                self.clip_actions,
            )
        actor_obs = self._last_obs["actor"]

        ref_joint_pos = actor_obs[:, :53]

        # [num_envs, 22]
        ref_hand_q = ref_joint_pos[:, self.hand_ids]

        # ---------------------------------------------------------
        # 4. 构造完整53维action
        # ---------------------------------------------------------
        full_actions = torch.zeros(
            self.num_envs,
            53,
            dtype=actions.dtype,
            device=self.device,
        )

        # 身体31维由PPO控制
        full_actions[:, self.body_ids] = actions
        full_actions[:, self.hand_ids] = ref_hand_q

        obs_dict, rew, terminated, truncated, extras = (
            self.env.step(full_actions)
        )

        term_or_trunc = terminated | truncated

        assert isinstance(rew, torch.Tensor)
        assert isinstance(term_or_trunc, torch.Tensor)

        dones = term_or_trunc.to(dtype=torch.long)

        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # ---------------------------------------------------------
        # 7. 转回rsl_rl需要的TensorDict
        # ---------------------------------------------------------
        obs = TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
        )

        # 保存下一步policy对应的observation
        self._last_obs = obs
        return (
            obs,
            rew,
            dones,
            extras,
        )