from collections import deque
from itertools import count

import dmc2gym
import gym

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


def initialize_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    
def reset_que(env, seq_len):
    aux_obs_que = deque(maxlen=seq_len+1)
    action_que = deque(maxlen=seq_len)
    rew_que = deque(maxlen=seq_len)
    done_que = deque(maxlen=seq_len)
    time_que = deque(maxlen=seq_len)
    
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # zero-padding
    for _ in range(seq_len):
        aux_obs_que.append(np.zeros(obs_shape))
        action_que.append(np.zeros(action_shape))
        
    return aux_obs_que, action_que, rew_que, done_que, time_que


def flatten(seq_tensor):
    B, S, D = seq_tensor.size()
    flat = seq_tensor.view(B, S*D)
    return flat


def calc_entropy(dist, squashed):
    squashed_ = torch.clamp(squashed, -1.0 + EPS, 1.0 - EPS)
    log_unsquashed = dist.log_prob(torch.atanh(squashed_))
    log_unsquashed = torch.clamp(log_unsquashed, -100, 100)
    log_det = torch.log(1 - torch.pow(squashed, 2) + EPS)
    entropy = (-log_unsquashed + log_det).mean(dim=-1)
    return entropy


def evaluate_agent(agent, args):
    if args.env == "Walker2d-v2":
        domain_name, task_name = "walker", "walk"
    else:
        raise NotImplementedError

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=None,
        visualize_reward=False,
        from_pixels=True,
        height=64,
        width=64,
        frame_skip=args.n_repeat,
    )

    agent.eval()

    cum_rew = 0.0
    aux_obs_que, action_que, *_ = reset_que(env, args.seq_len)
    obs = env.reset()
    aux_obs_que.append(obs)
    for t in count():
        aux_obs_seq = np.array(aux_obs_que)[1:]
        action_seq = np.array(action_que)[1:]
        converted_obs = torch.tensor(aux_obs_seq, dtype=torch.float32).div_(255.0).cuda()
        converted_action = torch.tensor(action_seq, dtype=torch.float32).cuda()

        S, C, H, W = converted_obs.size()
        _, A = converted_action.size()

        with torch.no_grad():
            aux_feature_seq = agent.slvm.calc_feature(converted_obs.view(1, S, C, H, W))
            action_seq = converted_action.view(1, S-1, A)
            _, aux_z_seq = agent.slvm.calc_latent(aux_feature_seq, action_seq)

            action = agent.actor(aux_z_seq[:,-1])[1]

        action = action.view(-1).to("cpu").numpy()

        next_obs, rew, done = env.step(action)[:3]

        cum_rew += rew

        if done or (t+1 == args.max_steps // args.n_repeat) or rew < 1e-6:
            break
        else:
            aux_obs_que.append(next_obs)
            action_que.append(action)

    return cum_rew
