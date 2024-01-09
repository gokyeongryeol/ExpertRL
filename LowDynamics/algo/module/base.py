import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import calc_entropy, initialize_weight


class MLP(nn.Module):
    def __init__(self, inp, hid, out, num_layer, activation=nn.ReLU(), dp_rate=0.0):
        super().__init__()

        unit = inp
        layer_list = []
        for _ in range(num_layer):
            layer_list.append(nn.Linear(unit, hid))
            layer_list.append(activation)
            unit = hid

        self.pre = nn.Sequential(*layer_list)
        self.post = nn.Linear(hid, out)

        self.apply(initialize_weight)

        self.dp_rate = dp_rate

    def forward(self, x):
        out = self.pre(x)

        if self.dp_rate > 0.0:
            out = F.dropout(out, p=self.dp_rate, training=True)

        out = self.post(out)
        return out


class Actor(nn.Module):
    def __init__(self, fea_dim, hid_dim, act_dim, seq_len,
                 num_layer, act_limit, min_log, max_log):
        super().__init__()

        self.actor = MLP(fea_dim*seq_len + act_dim*(seq_len-1),
                         hid_dim, act_dim * 2, num_layer)

        self.act_limit = act_limit
        self.min_log, self.max_log = min_log, max_log

    def forward(self, fa_seq):
        param = self.actor(fa_seq)
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
    def __init__(self, z_dim, hid_dim, act_dim, num_layer):
        super().__init__()

        self.critic = MLP(z_dim + act_dim, hid_dim, 1, num_layer)

    def forward(self, z, action):
        q_val = self.critic(torch.cat([z, action], dim=-1))
        return q_val
