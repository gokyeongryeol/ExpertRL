import math

import torch
import torch.nn as nn
import torch.optim as optim

from utils import calc_entropy

from .module import Actor, Critic

EPS = 1e-6
N_SAMPLE = 10


def hard_update(target, source):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(s_param.data)
        t_param.requires_grad = False


def soft_update(target, source, rho):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * rho + s_param.data * (1.0 - rho))


class SAC(nn.Module):
    def __init__(
        self,
        obs_dim,
        hid_dim,
        act_dim,
        q_layer,
        p_layer,
        act_limit,
        min_log,
        max_log,
        tunable,
        alpha,
        gamma,
        rho,
        q_lr=3e-4,
        p_lr=3e-4,
        p_wd=0.0,
    ):
        super().__init__()

        env_spec = (obs_dim, hid_dim, act_dim)
        self.critic1, self.c1_ema = Critic(*env_spec, q_layer), Critic(*env_spec, q_layer)
        hard_update(self.c1_ema, self.critic1)
        self.c1_optim = optim.Adam(self.critic1.parameters(), lr=q_lr)

        self.critic2, self.c2_ema = Critic(*env_spec, q_layer), Critic(*env_spec, q_layer)
        hard_update(self.c2_ema, self.critic2)
        self.c2_optim = optim.Adam(self.critic2.parameters(), lr=q_lr)

        self.actor = Actor(*env_spec, p_layer, act_limit, min_log, max_log)
        self.a_optim = optim.Adam(self.actor.parameters(), lr=p_lr, weight_decay=p_wd)

        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho

        self.tunable = tunable
        if tunable:
            self.log_alpha = nn.Parameter(torch.tensor([math.log(alpha + EPS)]))
            self.alpha_optim = optim.Adam([self.log_alpha], lr=p_lr)
            self.target_entropy = (-1) * act_dim

        self.upd_cnt = 0

    def min_double_q_trick(self, obs, action_hat, critic_lst):
        q_lst = []
        for critic in critic_lst:
            q_lst.append(critic(obs, action_hat))
        min_q = torch.min(torch.cat(q_lst, dim=-1), dim=-1).values
        return min_q

    @torch.no_grad()
    def calc_target(self, obs, next_obs, rew, done, off_agent=None):
        _, next_action_hat, next_entropy = self.actor(next_obs)
        critic_lst = [self.c1_ema, self.c2_ema]
        min_q = self.min_double_q_trick(next_obs, next_action_hat, critic_lst)

        if off_agent is not None:
            o_dist = off_agent.actor(next_obs)[0]
            next_cross_entropy = calc_entropy(
                o_dist, next_action_hat / self.actor.act_limit
            )
            bonus = next_entropy - next_cross_entropy
        else:
            bonus = next_entropy

        target = rew + self.gamma * (1 - done) * (min_q + self.alpha * bonus)
        return target

    def critic_update(self, obs, action, next_obs, rew, done, off_agent):
        c_loss_lst = []
        critic_lst = [self.critic1, self.critic2]
        optim_lst = [self.c1_optim, self.c2_optim]
        target = self.calc_target(obs, next_obs, rew, done, off_agent)
        for critic, c_optim in zip(critic_lst, optim_lst):
            q_value = critic(obs, action)
            c_loss = (torch.pow(q_value.squeeze(dim=-1) - target, 2)).mean()
            c_loss_lst.append(c_loss.item())

            c_optim.zero_grad()
            c_loss.backward()
            c_optim.step()

        return sum(c_loss_lst) / 2

    def actor_update(self, obs, action, off_agent=None):
        dist, action_hat, entropy = self.actor(obs)

        if off_agent is not None:
            o_dist = off_agent.actor(obs)[0]
            cross_entropy = calc_entropy(o_dist, action_hat / self.actor.act_limit)
            bonus = entropy - cross_entropy
        else:
            bonus = entropy

        critic_lst = [self.critic1, self.critic2]
        min_q = self.min_double_q_trick(obs, action_hat, critic_lst)
        a_loss = (-min_q - self.alpha * bonus).mean()

        self.a_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

        return a_loss.item()

    def alpha_update(self, obs):
        with torch.no_grad():
            entropy = self.actor(obs)[-1]

        alpha_loss = self.log_alpha * (entropy - self.target_entropy).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = torch.exp(self.log_alpha).item()

        return alpha_loss.item()

    def update_param(self, batch, off_agent=None):
        obs, action, next_obs, rew, done = batch

        c_loss = self.critic_update(obs, action, next_obs, rew, done, off_agent=off_agent)
        a_loss = self.actor_update(obs, action, off_agent=off_agent)

        if self.tunable:
            alpha_loss = self.alpha_update(obs)
        else:
            alpha_loss = 0.0

        soft_update(self.c1_ema, self.critic1, self.rho)
        soft_update(self.c2_ema, self.critic2, self.rho)

        self.upd_cnt += 1

        return {"c_loss": c_loss, "a_loss": a_loss, "alpha_loss": alpha_loss}
