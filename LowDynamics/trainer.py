import numpy as np
import torch
import torch.optim as optim

import dmc2gym
import gym

from itertools import count
from tqdm import tqdm
from utils import calc_entropy, evaluate_agent, flatten, reset_que


def train_offline_agent(agent, memory, args):
    metric_ = {"losses":
               {"KL": [], "NLL": []},
              "rewards": {
                  "eval": []}
              }

    metric = {
        "losses": {
            "s_loss": {"KL": [],
                       "NLL": []},
            "c_loss": [],
            "a_loss": [],
            "alpha_loss": [],
        },
        "rewards": {
            "eval": [],
        }
    }

    if args.from_scratch:
        for n_iters in tqdm(range(args.n_pretrain)):
            agent.train()

            m_batch = memory.sample(args.m_batch_size)
            loss_t = agent.update_param(m_batch, only_model=True)

            if (n_iters + 1) % args.n_period == 0:
                metric_["losses"]["KL"].append(loss_t["s_loss"]["KL"])
                metric_["losses"]["NLL"].append(loss_t["s_loss"]["NLL"])

                cum_rew = evaluate_agent(agent, args)
                metric_["rewards"]["eval"].append(cum_rew)

                torch.save(
                    agent.slvm.state_dict(),
                    f"ckpt/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt",
                )
                torch.save(
                    metric_,
                    f"metric/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt",
                )
    else:
        agent.slvm.load_state_dict(
            torch.load(f"ckpt/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt")
        )

    if not args.is_e2e:
        for param in agent.slvm.gen_z1_t.parameters():
            param.requires_grad_(True)

        agent.s_optim = optim.Adam(agent.slvm.parameters(), lr=1e-4)

    cum_rew = evaluate_agent(agent, args)
    metric["rewards"]["eval"].append(cum_rew)

    for n_iters in tqdm(range(args.n_offrl)):
        agent.train()

        m_batch = memory.sample(args.m_batch_size)
        ac_batch = memory.sample(args.ac_batch_size)
        loss_t = agent.update_param(m_batch, ac_batch, only_model=False)

        if (n_iters + 1) % args.n_period == 0:
            metric["losses"]["s_loss"]["KL"].append(loss_t["s_loss"]["KL"])
            metric["losses"]["s_loss"]["NLL"].append(loss_t["s_loss"]["NLL"])
            metric["losses"]["c_loss"].append(loss_t["c_loss"])
            metric["losses"]["a_loss"].append(loss_t["a_loss"])
            metric["losses"]["alpha_loss"].append(loss_t["alpha_loss"])

            cum_rew = evaluate_agent(agent, args)
            metric["rewards"]["eval"].append(cum_rew)

            torch.save(agent.state_dict(), f"ckpt/{args.env}_{args.alg}_{'orig' if args.is_e2e else 'plus'}_{args.seed}.pt")
            torch.save(metric, f"metric/{args.env}_{args.alg}_{'orig' if args.is_e2e else 'plus'}_{args.seed}.pt")
            print("Offline training: {} \t Reward: {}".format(n_iters, cum_rew))


def train_online_agent(act_dim, act_limit, agent, memory, args):
    metric_ = {"KL": [], "NLL": []}

    metric = {
        "losses": {
            "s_loss": {"KL": [],
                       "NLL": []},
            "c_loss": [],
            "a_loss": [],
            "alpha_loss": [],
        },
        "rewards": {
            "eval": [],
        }
    }

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

    n_steps = 0
    for episode in range(args.n_episodes):
        aux_obs_que, action_que, rew_que, done_que = reset_que(env, args.seq_len)
        obs = env.reset()

        aux_obs_que.append(obs)
        for t in count():
            agent.train()

            if n_steps < args.n_explore:
                squashed = np.tanh(2 * np.random.randn(act_dim), dtype=np.float32) 
                action = squashed * act_limit
            else:
                aux_obs_seq = np.array(aux_obs_que)[1:]
                action_seq = np.array(action_que)[1:]
                converted_obs = torch.tensor(aux_obs_seq, dtype=torch.float32).div_(255.0).cuda()
                converted_action = torch.tensor(action_seq, dtype=torch.float32).cuda()

                S, C, H, W = converted_obs.size()
                _, A = converted_action.size()

                with torch.no_grad():
                    aux_feature_seq = agent.slvm.encoder(converted_obs.view(1, S, C, H, W))
                    action_seq = converted_action.view(1, S-1, A)
                    fa_seq = torch.cat([flatten(aux_feature_seq), flatten(action_seq)], dim=-1)
                    action = agent.actor(fa_seq)[1]

                action = action.view(-1).to("cpu").numpy()

            next_obs, rew, done = env.step(action)[:3]
            terminate = done or (t == args.max_steps // args.n_repeat) or rew < 1e-6

            aux_obs_que.append(next_obs)
            action_que.append(action)
            rew_que.append(rew)
            done_que.append(terminate)

            if t+1 >= args.seq_len:
                aux_obs_t = np.array(aux_obs_que)
                action_t = np.array(action_que)
                rew_t = np.array(rew_que)
                done_t = np.array(done_que)

                memory.push(aux_obs_t, action_t, rew_t, done_t)

            n_steps += 1

            if n_steps == args.n_explore:
                if args.from_scratch:
                    for n_iters in tqdm(range(args.n_pretrain)):
                        m_batch = memory.sample(args.m_batch_size)
                        loss_t = agent.update_param(m_batch, only_model=True)

                        if (n_iters + 1) % args.n_period == 0:
                            metric_["KL"].append(loss_t["s_loss"]["KL"])
                            metric_["NLL"].append(loss_t["s_loss"]["NLL"])

                            torch.save(
                                agent.slvm.state_dict(),
                                f"ckpt/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt",
                            )
                            torch.save(
                                metric_,
                                f"metric/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt",
                            )

                else:
                    agent.slvm.load_state_dict(
                        torch.load(
                            f"ckpt/{args.env}_{args.alg}_slvm_{'orig' if args.is_e2e else 'plus'}.pt",
                        ),
                    )

                if not args.is_e2e:
                    for param in agent.slvm.gen_z1_t.parameters():
                            param.requires_grad_(True)

                    agent.s_optim = optim.Adam(agent.slvm.parameters(), lr=1e-4)

            if n_steps > args.n_explore:
                m_batch = memory.sample(args.m_batch_size)
                ac_batch = memory.sample(args.ac_batch_size)
                loss_t = agent.update_param(m_batch, ac_batch, only_model=False)

                if (n_steps + 1) % args.n_period == 0:
                    metric["losses"]["s_loss"]["KL"].append(loss_t["s_loss"]["KL"])
                    metric["losses"]["s_loss"]["NLL"].append(loss_t["s_loss"]["NLL"])
                    metric["losses"]["c_loss"].append(loss_t["c_loss"])
                    metric["losses"]["a_loss"].append(loss_t["a_loss"])
                    metric["losses"]["alpha_loss"].append(loss_t["alpha_loss"])

            if n_steps % args.n_period == 0:
                eval_cum_rew = evaluate_agent(agent, args)
                metric["rewards"]["eval"].append(eval_cum_rew)
                print(
                    "Episode: {} \t Steps : {} \t Eval reward: {}".format(
                        episode, n_steps, eval_cum_rew,
                    )
                )
                torch.save(
                    agent.state_dict(), f"ckpt/{args.env}_{args.alg}_{'orig' if args.is_e2e else 'plus'}_{args.seed}.pt"
                )
                torch.save(
                    metric, f"metric/{args.env}_{args.alg}_{'orig' if args.is_e2e else 'plus'}_{args.seed}.pt"
                )

            if terminate:
                break

        if n_steps >= 1e6:
            break
