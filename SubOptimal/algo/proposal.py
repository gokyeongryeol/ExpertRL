import torch
import torch.nn.functional as F

from .base import SAC


class TD_Ratio(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_beta(self, beta):
        self.beta = beta

    @torch.no_grad()
    def estimate_reward(self, obs, action, next_obs, done, agent=None):
        act_limit = self.actor.act_limit

        if agent is None:
            agent = self

        critic_lst = [agent.critic1, agent.critic2]
        c_min_qs = [critic(obs, action) for critic in critic_lst]
        dist, next_action_hat, next_entropy = agent.actor(next_obs)

        ema_lst = [agent.c1_ema, agent.c2_ema]
        n_min_q = self.min_double_q_trick(
            next_obs, next_action_hat, ema_lst,
        )

        rew_ests = torch.stack([
            c_min_q.squeeze() - self.gamma * (1 - done) * (n_min_q + self.alpha * next_entropy)
            for c_min_q in c_min_qs
        ])
        return rew_ests

    @torch.no_grad()
    def compute_weight(self, obs, action, next_obs, rew, done, off_agent):
        o_rew_ests = self.estimate_reward(obs, action, next_obs, done, off_agent)
        rew_ests = self.estimate_reward(obs, action, next_obs, done)

        o_avg_gap = torch.stack(
            [abs(rew - o_rew_est) for o_rew_est in o_rew_ests]
        ).mean(dim=0)

        avg_gap = torch.stack(
            [abs(rew - rew_est) for rew_est in rew_ests]
        ).mean(dim=0)

        weight = F.softmax(self.beta * avg_gap / o_avg_gap, dim=0)
        return weight

    def critic_update(self, obs, action, next_obs, rew, done, off_agent):
        weight = self.compute_weight(obs, action, next_obs, rew, done, off_agent)

        c_loss_lst = []
        critic_lst = [self.critic1, self.critic2]
        optim_lst = [self.c1_optim, self.c2_optim]
        target = self.calc_target(obs, next_obs, rew, done, off_agent)
        for critic, c_optim in zip(critic_lst, optim_lst):
            q_value = critic(obs, action)
            c_loss = (len(obs) * weight * torch.pow(q_value.squeeze(dim=-1) - target, 2)).mean()
            c_loss_lst.append(c_loss.item())

            c_optim.zero_grad()
            c_loss.backward()
            c_optim.step()

        return sum(c_loss_lst) / 2
