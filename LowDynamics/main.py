import argparse
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn

import gym

from agent import make_offline_agent, make_online_agent
from trainer import train_offline_agent, train_online_agent


def make_dirs():
    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("metric", exist_ok=True)


def set_random_seed(rs):
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(rs)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env", default="Walker2d-v2", type=str, help="name of the environment",
    )
    parser.add_argument("--seed", default=3061, type=int, help="random seed")

    parser.add_argument("--use_offline", action="store_true")
    parser.add_argument("--is_e2e", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--alg", default="slac", type=str)

    parser.add_argument(
        "--memory_size", default=int(1e5), type=int, help="capacity of replay buffer"
    )
    parser.add_argument(
        "--n_episodes", default=int(1e6), type=int, help="number of episodes"
    )
    parser.add_argument(
        "--n_explore",
        default=int(1e4),
        type=int,
        help="number of rollouts before training",
    )
    parser.add_argument(
        "--n_period",
        default=int(1e2),
        type=int,
        help="number of iterations or steps between logging",
    )
    parser.add_argument(
        "--n_pretrain",
        default=int(1e5),
        type=int,
        help="number of pretraining iterations for slvm",
    )
    parser.add_argument(
        "--n_offrl",
        default=int(5e5),
        type=int,
        help="number of rollouts before training",
    )
    parser.add_argument(
        "--max_steps",
        default=int(1e3),
        type=int,
        help="number of maximum steps within a episode",
    )
    parser.add_argument(
        "--m_batch_size",
        default=32,
        type=int,
        help="batch size for slvm update",
    )
    parser.add_argument(
        "--ac_batch_size",
        default=512,
        type=int,
        help="batch size for actor-critic update",
    )
    parser.add_argument(
        "--seq_len",
        default=8,
        type=int,
        help="sequence length for pomdp setting",
    )
    args = parser.parse_args()

    return args


def off_rl(obs_dim, act_dim, act_limit, args):
    off_agent, off_memory = make_offline_agent(obs_dim, act_dim, act_limit, args)
    train_offline_agent(off_agent, off_memory, args)


def on_rl(obs_dim, act_dim, act_limit, args):
    on_agent, on_memory = make_online_agent(obs_dim, act_dim, act_limit, args)
    train_online_agent(act_dim, act_limit, on_agent, on_memory, args)


if __name__ == "__main__":
    make_dirs()
    args = get_args_parser()

    set_random_seed(args.seed)

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    env_spec = (obs_dim, act_dim, act_limit)

    if args.use_offline:
        off_rl(*env_spec, args)
    else:
        on_rl(*env_spec, args)
