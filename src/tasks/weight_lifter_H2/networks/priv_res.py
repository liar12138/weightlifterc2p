
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import *
from src.networks.simple_cnn import DepthCNN,DepthCNN2X

class PrivResMLP(MLPModel):

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims = (256, 256, 256), activation = "elu", obs_normalization = False, distribution_cfg = None):
        dcf = copy.deepcopy(distribution_cfg)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        self.expert = MLPModel(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, dcf)
        self.expert.load_state_dict(torch.load('src/tasks/weight_lifter_H2/networks/experts/model_16700.pt')['actor_state_dict'])
        for pm in self.expert.parameters():
            pm.requires_grad = False
        self.barbell_dim=obs["barbell"].shape[-1]
        self.priv_normalizer = EmpiricalNormalization(self.barbell_dim)
        self.layer_norm=torch.nn.LayerNorm(53)

    def get_latent(self, obs, masks = None, hidden_state = None):
        latent = super().get_latent(obs, masks, hidden_state)
        eo = self.expert(obs,masks,hidden_state)
        priv = self.priv_normalizer(obs["barbell"])
        return torch.cat([latent,priv] ,dim=-1), eo

    def update_normalization(self, obs):
        self.priv_normalizer.update(obs["barbell"])  # type: ignore
        return super().update_normalization(obs)

    def _get_latent_dim(self):
        return super()._get_latent_dim() + 53 + 16

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent,expert_output = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(torch.cat([latent, self.layer_norm(expert_output)],dim=-1)) + expert_output

        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
    
        return mlp_output