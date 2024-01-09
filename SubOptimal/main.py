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
        "--env", default="Walker2d-v2", type=str, help="name of the environment"
    )
    parser.add_argument("--seed", default=3061, type=int, help="random seed")
    parser.add_argument("--use_offline", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--offline_alg", default="AWAC", type=str)
    parser.add_argument("--online_alg", default="TD_Ratio", type=str)

    parser.add_argument(
        "--n_episodes", default=int(1e6), type=int, help="number of episodes"
    )
    parser.add_argument(
        "--n_explore",
        default=int(1e4),
        type=int,
        help="number of rollouts before training",
    )
    parser.add_argument("--n_period", default=int(1e3), type=int, help="number of iterations or steps between evaluation")
    parser.add_argument(
        "--n_bc",
        default=int(5e4),
        type=int,
        help="number of behavior cloning iterations to offline dataset",
    )
    parser.add_argument(
        "--n_offrl",
        default=int(5e5),
        type=int,
        help="number of pretraining iterations to offline dataset",
    )
    parser.add_argument(
        "--max_steps",
        default=int(1e3),
        type=int,
        help="number of maximum steps within a episode",
    )

    parser.add_argument(
        "--hid_dim", default=256, type=int, help="number of hidden units"
    )

    parser.add_argument("--beta", default=0.01, type=float, help="smoothing scale")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--rho", default=0.995, type=float, help="ema coefficient")

    args = parser.parse_args()

    if not args.use_offline:
        alg = args.online_alg
    else:
        if "TD_Ratio" in args.online_alg:
            alg = f"{args.offline_alg}_{args.online_alg}_{args.beta}"
        else:
            alg = f"{args.offline_alg}_{args.online_alg}"

    return args


def on_rl(obs_dim, act_dim, act_limit, args):
    on_agent, on_memory = make_online_agent(obs_dim, act_dim, act_limit, args)
    train_online_agent(act_dim, act_limit, on_agent, on_memory, args)


def kl_reg_rl(obs_dim, act_dim, act_limit, args):
    if args.from_scratch:
        off_agent, off_memory = make_offline_agent(obs_dim, act_dim, act_limit, args)
        train_offline_agent(off_agent, off_memory, args)
    else:
        off_agent, _ = make_offline_agent(obs_dim, act_dim, act_limit, args)
    off_agent.load_state_dict(torch.load(f"ckpt/{args.env}_{args.offline_alg}.pt"))
    off_agent.eval()

    on_agent, on_memory = make_online_agent(obs_dim, act_dim, act_limit, args)
    train_online_agent(act_dim, act_limit, on_agent, on_memory, args, off_agent)


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
        kl_reg_rl(*env_spec, args)
    else:
        on_rl(*env_spec, args)
