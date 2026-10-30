from rsl_rl.models.mlp_model import *
from rsl_rl.modules import CNN, MLP
class BodyMLPWithFixedHands(MLPModel):
    """
    Actor只输出31维身体action，
    手部22维从command reference中直接生成，
    最终返回53维action给环境。
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    ):
        # PPO actor只学习31维
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=31,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.body_ids = torch.tensor(
            [
                0,1,2,3,4,5,
                6,7,8,9,10,11,
                12,13,14,
                15,16,
                17,18,19,20,21,22,23,
                35,36,37,38,39,40,41,
            ],
            dtype=torch.long,
        )

        self.hand_ids = torch.tensor(
            [
                24,25,26,27,28,29,30,31,32,33,34,
                42,43,44,45,46,47,48,49,50,51,52,
            ],
            dtype=torch.long,
        )

    def forward(
        self,
        obs,
        masks=None,
        hidden_state=None,
        stochastic_output=False,
    ):

        # 1. 网络只输出31维身体action
        body_action = super().forward(
            obs,
            masks,
            hidden_state,
            stochastic_output,
        )

        ref_joint_pos = obs["actor"][:, :53]

        hand_ids = self.hand_ids.to(ref_joint_pos.device)

        ref_hand_q = ref_joint_pos[:, hand_ids]

        # 3. 这里先生成最终53维
        full_action = torch.zeros(
            body_action.shape[0],
            53,
            device=body_action.device,
            dtype=body_action.dtype,
        )

        body_ids = self.body_ids.to(body_action.device)

        full_action[:, body_ids] = body_action

        # 注意：
        # 这里不能一定直接放ref_hand_q，
        # 取决于你的JointPositionAction是否还有scale/offset
        full_action[:, hand_ids] = ref_hand_q

        return full_action