from itertools import count

import gym
import numpy as np
import torch

from utils import calc_entropy, evaluate_agent

BC_BATCH_SIZE = 128
BATCH_SIZE = {"AWAC": 1024, "SAC": 256, "TD_Ratio": 256}


def train_offline_agent(off_agent, off_memory, args):
    off_metric = {
        "losses": {
            "c_loss": [],
            "a_loss": [],
            "alpha_loss": [],
        },
        "rewards": {
            "offline": []
        }
    }
            
    for t in range(args.n_bc + args.n_offrl):
        off_agent.train()

        off_batch = off_memory.sample(
            BATCH_SIZE[args.offline_alg] if t >= args.n_bc else BC_BATCH_SIZE
        )
        off_loss_t = off_agent.update_param(off_batch)

        off_metric["losses"]["c_loss"].append(off_loss_t["c_loss"])
        off_metric["losses"]["a_loss"].append(off_loss_t["a_loss"])
        off_metric["losses"]["alpha_loss"].append(off_loss_t["alpha_loss"])

        if (t + 1) % args.n_period == 0:
            cum_rew = evaluate_agent(off_agent, args)
            off_metric["rewards"]["offline"].append(cum_rew)

            torch.save(off_agent.state_dict(), f"ckpt/{args.env}_{args.offline_alg}.pt")
            torch.save(off_metric, f"metric/{args.env}_{args.offline_alg}.pt")
            print("Pre-training: {} \t Reward: {}".format(t, cum_rew))


def train_online_agent(act_dim, act_limit, on_agent, on_memory, args, off_agent=None):
    on_metric = {
        "losses": {
            "c_loss": [],
            "a_loss": [],
            "alpha_loss": [],
        },
        "rewards": {
            "online-eval": [],
        }
    }

    if off_agent is None:
        alg = args.online_alg
    else:
        if "TD_Ratio" in args.online_alg:
            alg = f"{args.offline_alg}_{args.online_alg}_{args.beta}"
        else:
            alg = f"{args.offline_alg}_{args.online_alg}"

    env = gym.make(args.env)

    n_steps = 0
    for episode in range(args.n_episodes):
        obs = env.reset()[0]
        for t in count():
            on_agent.train()

            if n_steps < args.n_explore:
                squashed = np.tanh(2 * np.random.randn(act_dim), dtype=np.float32) 
                action = squashed * act_limit
            else:
                obs_ = torch.tensor(obs, dtype=torch.float32).view(1, -1)
                obs_ = obs_.cuda()

                with torch.no_grad():
                    action_ = on_agent.actor(obs_)[1]

                action = action_.view(-1).to("cpu").numpy()

            next_obs, rew, done = env.step(action)[:3]
            on_memory.push(obs, action, next_obs, rew, done)

            terminate = done or ((t + 1) == args.max_steps)

            n_steps += 1
            if n_steps > args.n_explore:
                on_batch = on_memory.sample(BATCH_SIZE[args.online_alg])
                on_metric_t = on_agent.update_param(on_batch, off_agent=off_agent)

                on_metric["losses"]["c_loss"].append(on_metric_t["c_loss"])
                on_metric["losses"]["a_loss"].append(on_metric_t["a_loss"])
                on_metric["losses"]["alpha_loss"].append(on_metric_t["alpha_loss"])

            if n_steps % args.n_period == 0:
                eval_cum_rew = evaluate_agent(on_agent, args)
                on_metric["rewards"]["online-eval"].append(eval_cum_rew)
                print(
                    "Episode: {} \t Steps : {} \t Eval reward: {}".format(
                        episode, n_steps, eval_cum_rew,
                    )
                )
                torch.save(on_agent.state_dict(), f"ckpt/{args.env}_{alg}_{args.seed}.pt")
                torch.save(
                    on_metric, f"metric/{args.env}_{alg}_{args.seed}.pt"
                )

            if terminate:
                break
            else:
                obs = next_obs

        if n_steps >= 1e6:
            break