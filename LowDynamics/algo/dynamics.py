import torch
import torch.nn as nn
from .module import MLP


class Dynamics(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, num_layer, dp_rate=0.1, n_MCdrop=10):
        super().__init__()

        self.net = MLP(obs_dim + act_dim, hid_dim, obs_dim, num_layer, dp_rate=dp_rate)
        self.n_MCdrop = n_MCdrop

    def forward(self, obs, action, compute_std=False):
        concat = torch.cat([obs, action], dim=-1)
        next_obs_lst = [self.net(concat) for _ in range(self.n_MCdrop)]
        next_obses = torch.stack(next_obs_lst)

        pred = next_obses.mean(dim=0)
        if compute_std:
            std = next_obses.std(dim=0)
            return pred, std

        return pred
