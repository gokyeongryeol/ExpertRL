import torch
import torch.nn as nn
from .module import MLP
from torch.distributions import Normal


class Dynamics(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, num_layer):
        super().__init__()

        self.net = MLP(obs_dim + act_dim, hid_dim, obs_dim*2, num_layer)

    def forward(self, obs, action):
        param = self.net(torch.cat([obs, action], dim=-1))
        mu, omega = torch.chunk(param, chunks=2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(omega)

        dist = Normal(mu, sigma)
        return dist
