import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from utils import calc_entropy


class MLP(nn.Module):
    def __init__(self, inp, hid, out, num_layer):
        super().__init__()

        unit = inp
        layer_list = []
        for _ in range(num_layer):
            bound = 1. / np.sqrt(hid)
            linear = nn.Linear(unit, hid)
            linear.weight.data.uniform_(-bound, bound)
            linear.bias.data.fill_(0)
            
            layer_list.append(linear)
            layer_list.append(nn.ReLU())
            unit = hid

        bound = 3e-3
        linear = nn.Linear(hid, out)
        linear.weight.data.uniform_(-bound, bound)
        linear.bias.data.fill_(0)

        layer_list.append(linear)
        self.mlp = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.mlp(x)
        return x


class Actor(nn.Module):
    def __init__(
        self, obs_dim, hid_dim, act_dim, num_layer, act_limit, min_log, max_log
    ):
        super().__init__()

        self.actor = MLP(obs_dim, hid_dim, act_dim * 2, num_layer)

        self.act_limit = act_limit
        self.min_log, self.max_log = min_log, max_log

    def forward(self, obs):
        param = self.actor(obs)
        mu, omega = torch.chunk(param, chunks=2, dim=-1)
        log_sig = torch.clamp(omega, self.min_log, self.max_log)
        sigma = torch.exp(log_sig) + 1e-6

        dist = Normal(mu, sigma)
        if self.training:
            unsquashed = dist.rsample()
        else:
            unsquashed = dist.mean

        squashed = torch.tanh(unsquashed)
        action_hat = squashed * self.act_limit
        entropy = calc_entropy(dist, squashed)
        return dist, action_hat, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, num_layer):
        super().__init__()

        self.critic = MLP(obs_dim + act_dim, hid_dim, 1, num_layer)

    def forward(self, obs, action):
        q_val = self.critic(torch.cat([obs, action], dim=-1))
        return q_val
