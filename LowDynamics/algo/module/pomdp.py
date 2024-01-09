import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from utils import initialize_weight


class Encoder(nn.Module):
    def __init__(self, fea_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(32, 64, 3, 2, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(64, 128, 3, 2, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(128, 256, 3, 2, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(256, fea_dim, 4),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    ).apply(initialize_weight)

    def forward(self, obs_seq):
        B, S, C, H, W = obs_seq.size()
        obs_seq = obs_seq.view(B * S, C, H, W)
        feature_seq = self.encoder(obs_seq)
        feature_seq = feature_seq.view(B, S, -1)
        return feature_seq
    

class Decoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, std=math.sqrt(0.1)):
        super().__init__()
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(z1_dim+z2_dim, 256, 4),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    ).apply(initialize_weight)

        self.std = std
        
    def forward(self, aux_z_seq):
        B, _, z_dim = aux_z_seq.size()
        aux_z_seq = aux_z_seq.view(-1, z_dim, 1, 1)
        mu = self.decoder(aux_z_seq)
        _, C, H, W = mu.size()
        mu = mu.view(B, -1, C, H, W)
        sigma = self.std * torch.ones(mu.size()).cuda()
        
        dist = Normal(mu, sigma)
        if self.training:
            aux_obs_seq_hat = dist.rsample()
        else:
            aux_obs_seq_hat = dist.mean
        return dist, aux_obs_seq_hat

    
class Prior(nn.Module):
    def __init__(self, z1_dim, std=1.0):
        super().__init__()
        self.z1_dim = z1_dim
        self.std = std

    def forward(self, batch_size):
        mu = torch.zeros(batch_size, self.z1_dim).cuda()
        sigma = self.std * torch.ones(mu.size()).cuda()

        dist = Normal(mu, sigma)
        return dist
