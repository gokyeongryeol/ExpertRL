import torch
import torch.nn.functional as F

from utils import calc_entropy
from .base import SAC


class AWAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actor_update(self, obs, action, off_agent=None):
        dist, action_hat, entropy = self.actor(obs)
        cross_entropy = calc_entropy(dist, action / self.actor.act_limit)

        with torch.no_grad():
            critic_lst = [self.critic1, self.critic2]
            min_q_avg = self.min_double_q_trick(obs, action_hat, critic_lst)
            min_q_curr = self.min_double_q_trick(obs, action, critic_lst)
            adv = min_q_curr - min_q_avg

            weight = F.softmax(adv, dim=0)

        adv_cross_entropy = len(adv) * weight * cross_entropy
        a_loss = (adv_cross_entropy - self.alpha * entropy).mean()

        self.a_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

        return a_loss.item()
