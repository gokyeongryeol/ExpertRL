import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import calc_entropy, flatten

from .module import Actor, Critic, SLVM

EPS = 1e-6


def hard_update(target, source):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(s_param.data)
        t_param.requires_grad = False


def soft_update(target, source, rho):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * rho + s_param.data * (1.0 - rho))


class SLAC(nn.Module):
    def __init__(
        self,
        slvm,
        obs_dim,
        act_dim,
        act_limit,
        seq_len,
        hid_dim=256,
        fea_dim=256,
        z2_dim=256,
        q_layer=2,
        p_layer=2,
        min_log=-20,
        max_log=2,
        tunable=True,
        alpha=1.0,
        gamma=0.99,
        rho=0.995,
        s_lr=1e-4,
        q_lr=3e-4,
        p_lr=3e-4,
        p_wd=0.0,
    ):
        super().__init__()

        self.slvm = slvm
        self.s_optim = optim.Adam(self.slvm.parameters(), lr=s_lr)

        c_spec = (obs_dim+z2_dim, hid_dim, act_dim)
        self.critic1, self.c1_ema = Critic(*c_spec, q_layer), Critic(*c_spec, q_layer)
        hard_update(self.c1_ema, self.critic1)
        self.c1_optim = optim.Adam(self.critic1.parameters(), lr=q_lr)

        self.critic2, self.c2_ema = Critic(*c_spec, q_layer), Critic(*c_spec, q_layer)
        hard_update(self.c2_ema, self.critic2)
        self.c2_optim = optim.Adam(self.critic2.parameters(), lr=q_lr)

        a_spec = (fea_dim, hid_dim, act_dim, seq_len)
        self.actor = Actor(*a_spec, p_layer, act_limit, min_log, max_log)
        self.a_optim = optim.Adam(self.actor.parameters(), lr=p_lr, weight_decay=p_wd)

        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho

        self.tunable = tunable
        if tunable:
            self.log_alpha = nn.Parameter(torch.tensor([math.log(alpha + EPS)]))
            self.alpha_optim = optim.Adam([self.log_alpha], lr=p_lr)
            self.target_entropy = (-1) * act_dim

    def slvm_update(self, aux_obs_seq, action_seq, rew_seq, done_seq):
        aux_feature_seq = self.slvm.calc_feature(aux_obs_seq)
        KL, aux_z_seq = self.slvm.calc_latent(aux_feature_seq, action_seq)
        NLL, _ = self.slvm.reconstruct(aux_obs_seq, aux_z_seq, action_seq, rew_seq, done_seq)
        KL, NLL = KL.mean(), NLL.mean()

        self.s_optim.zero_grad()
        (KL + NLL).backward()
        self.s_optim.step()

        return KL.item(), NLL.item()

    def min_double_q_trick(self, z, action_hat, critic_lst):
        q_lst = []
        for critic in critic_lst:
            q_lst.append(critic(z, action_hat))
        min_q = torch.min(torch.cat(q_lst, dim=-1), dim=-1).values
        return min_q

    @torch.no_grad()
    def calc_target(self, next_fa_seq, next_z, rew, done):
        _, next_action_hat, next_entropy = self.actor(next_fa_seq)
        critic_lst = [self.c1_ema, self.c2_ema]
        min_q = self.min_double_q_trick(next_z, next_action_hat, critic_lst)

        target = rew + self.gamma * (1 - done) * (min_q + self.alpha * next_entropy)
        return target

    def critic_update(self, z, action, next_fa_seq, next_z, rew, done):
        c_loss_lst = []
        critic_lst = [self.critic1, self.critic2]
        optim_lst = [self.c1_optim, self.c2_optim]
        target = self.calc_target(next_fa_seq, next_z, rew, done)
        for critic, c_optim in zip(critic_lst, optim_lst):
            q_value = critic(z, action)
            c_loss = (torch.pow(q_value.squeeze(dim=-1) - target, 2)).mean()
            c_loss_lst.append(c_loss.item())

            c_optim.zero_grad()
            c_loss.backward()
            c_optim.step()

        return sum(c_loss_lst) / 2

    def actor_update(self, fa_seq, z, action):
        dist, action_hat, entropy = self.actor(fa_seq)
        cross_entropy = calc_entropy(dist, action / self.actor.act_limit)

        with torch.no_grad():
            critic_lst = [self.critic1, self.critic2]
            min_q_avg = self.min_double_q_trick(z, action_hat, critic_lst)
            min_q_curr = self.min_double_q_trick(z, action, critic_lst)
            adv = min_q_curr - min_q_avg

            weight = F.softmax(adv, dim=0)

        adv_cross_entropy = len(adv) * weight * cross_entropy
        a_loss = (adv_cross_entropy - self.alpha * entropy).mean()

        self.a_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

        return a_loss.item()

    def alpha_update(self, fa_seq):
        with torch.no_grad():
            entropy = self.actor(fa_seq)[-1]

        alpha_loss = self.log_alpha * (entropy - self.target_entropy).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = torch.exp(self.log_alpha).item()

        return alpha_loss.item()

    def update_param(self, m_batch, ac_batch=None, only_model=False):
        aux_obs_seq, action_seq, rew_seq, done_seq = m_batch
        KL, NLL = self.slvm_update(aux_obs_seq, action_seq, rew_seq, done_seq)

        c_loss, a_loss, alpha_loss = 0.0, 0.0, 0.0
        if not only_model:
            aux_obs_seq, action_seq, rew_seq, done_seq = ac_batch
            with torch.no_grad():
                aux_feature_seq = self.slvm.calc_feature(aux_obs_seq)
                _, aux_z_seq, rew_penalty = self.slvm.calc_latent(aux_feature_seq, action_seq, return_sigma=True)

            pre_feature, post_feature = aux_feature_seq[:,:-1], aux_feature_seq[:,1:]
            pre_action, post_action = action_seq[:,:-1], action_seq[:,1:]
            fa_seq = torch.cat([flatten(pre_feature), flatten(pre_action)], dim=-1)
            next_fa_seq = torch.cat([flatten(post_feature), flatten(post_action)], dim=-1)

            z, action, next_z = aux_z_seq[:,-2], action_seq[:,-1], aux_z_seq[:,-1] 
            rew, done = rew_seq[:,-1] - 5.0 * rew_penalty, done_seq[:,-1]

            c_loss = self.critic_update(z, action, next_fa_seq, next_z, rew, done)
            a_loss = self.actor_update(fa_seq, z, action)

            if self.tunable:
                alpha_loss = self.alpha_update(fa_seq)
            else:
                alpha_loss = 0.0

            soft_update(self.c1_ema, self.critic1, self.rho)
            soft_update(self.c2_ema, self.critic2, self.rho)

        return {
            "s_loss": {"KL": KL, "NLL": NLL},
            "c_loss": c_loss,
            "a_loss": a_loss,
            "alpha_loss": alpha_loss
        }
