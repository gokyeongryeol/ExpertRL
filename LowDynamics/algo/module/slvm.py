import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, kl_divergence
from .base import MLP
from .pomdp import Encoder, Decoder, Prior

EPS = 1e-6


class SLVM(nn.Module):
    def __init__(self, fea_dim, hid_dim, z1_dim, z2_dim, act_dim, num_layer=2, dynamics=None, estimate_rew=False):
        super().__init__() 

        activation = nn.LeakyReLU(0.2)

        self.encoder = Encoder(fea_dim)
        self.decoder = Decoder(z1_dim, z2_dim)

        self.gen_z1_1 = Prior(z1_dim)
        self.inf_z1_1 = MLP(fea_dim, hid_dim, z1_dim*2, num_layer, activation=activation)
        self.gen_z2_1 = MLP(fea_dim+z1_dim, hid_dim, z2_dim*2, num_layer, activation=activation)

        self.use_dynamics = dynamics is not None
        if self.use_dynamics:
            self.gen_z1_t = dynamics
        else:
            self.gen_z1_t = MLP(z1_dim+act_dim, hid_dim, z1_dim*2, num_layer, activation=activation)

        self.inf_z1_t = MLP(z1_dim+act_dim+fea_dim, hid_dim, z1_dim*2, num_layer, activation=activation)
        self.gen_z2_t = MLP(z2_dim+act_dim+fea_dim+z1_dim, hid_dim, z2_dim*2, num_layer, activation=activation)

        self.estimate_rew = estimate_rew
        if self.estimate_rew:
            self.gen_rew = MLP((z1_dim+z2_dim)*2+act_dim, hid_dim, 1*2, num_layer, activation=activation)

    def calc_feature(self, aux_obs_seq):
        aux_feature_seq = self.encoder(aux_obs_seq)
        return aux_feature_seq

    def calc_dist(self, param):
        mu, log_sig = torch.chunk(param, chunks=2, dim=-1)
        sigma = F.softplus(log_sig) + EPS

        mu, sigma = mu.squeeze(dim=-1), sigma.squeeze(dim=-1)
        dist = Normal(mu, sigma)
        if self.training:
            pred = dist.rsample()
        else:
            pred = dist.mean
        return dist, pred

    def calc_latent(self, aux_feature_seq, action_seq, return_sigma=False):
        KL_lst = []
        z_t_list = []
        feature = aux_feature_seq[:,0]

        batch_size = feature.size(0)
        prior = self.gen_z1_1(batch_size)

        z1_t_param = self.inf_z1_1(feature)
        z1_t_dist, z1_t = self.calc_dist(z1_t_param)
        KL_lst.append(kl_divergence(z1_t_dist, prior).mean(dim=-1))

        z2_t_param = self.gen_z2_1(torch.cat([feature, z1_t], dim=-1))
        _, z2_t = self.calc_dist(z2_t_param)

        z_t = torch.cat([z1_t, z2_t], dim=-1)
        z_t_list.append(z_t)

        for t in range(1, aux_feature_seq.size(1)):
            next_feature, action = aux_feature_seq[:,t], action_seq[:,t-1]

            if self.use_dynamics:
                # mu, std = self.gen_z1_t(z1_t, action, compute_std=True)
                # base_std = math.sqrt(0.3) * torch.ones(std.size()).to(std.device)

                # prior = Normal(mu, base_std + std)
                prior = self.gen_z1_t(z1_t, action)
            else:
                prior_param = self.gen_z1_t(torch.cat([z1_t, action], dim=-1))
                prior, _ = self.calc_dist(prior_param)

            z1_t_param = self.inf_z1_t(torch.cat([z1_t, action, next_feature], dim=-1))
            z1_t_dist, z1_t = self.calc_dist(z1_t_param)
            KL_lst.append(kl_divergence(z1_t_dist, prior).mean(dim=-1))

            z2_t_param = self.gen_z2_t(torch.cat([z2_t, action, next_feature, z1_t], dim=-1))
            _, z2_t = self.calc_dist(z2_t_param)

            z_t = torch.cat([z1_t, z2_t], dim=-1)
            z_t_list.append(z_t)

        KL = torch.stack(KL_lst).mean(dim=0)
        aux_z_seq = torch.stack(z_t_list, dim=1)

        if self.estimate_rew and return_sigma:
            zaz_seq = torch.cat([aux_z_seq[:,:-1], action_seq, aux_z_seq[:,1:]], dim=-1)
            rew_param = self.gen_rew(zaz_seq)
            rew_seq_dist, _ = self.calc_dist(rew_param)

            return KL, aux_z_seq, rew_seq_dist.stddev[:,-1]

        return KL, aux_z_seq

    def reconstruct(self, aux_obs_seq, aux_z_seq, action_seq, rew_seq, done_seq):
        aux_obs_seq_dist, aux_obs_seq_hat = self.decoder(aux_z_seq)
        NLL = - aux_obs_seq_dist.log_prob(aux_obs_seq).mean(dim=(1,2,3,4))

        if self.estimate_rew:
            zaz_seq = torch.cat([aux_z_seq[:,:-1], action_seq, aux_z_seq[:,1:]], dim=-1)
            rew_param = self.gen_rew(zaz_seq)
            rew_seq_dist, _ = self.calc_dist(rew_param)
            NLL = NLL - ((1-done_seq) * rew_seq_dist.log_prob(rew_seq)).mean(dim=-1)
        return NLL, aux_obs_seq_hat
