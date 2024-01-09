from itertools import count
import gym
import torch

EPS = 1e-6


def calc_entropy(dist, squashed):
    squashed_ = torch.clamp(squashed, -1.0 + EPS, 1.0 - EPS)
    log_unsquashed = dist.log_prob(torch.atanh(squashed_))
    log_unsquashed = torch.clamp(log_unsquashed, -100, 100)
    log_det = torch.log(1 - torch.pow(squashed, 2) + EPS)
    entropy = (-log_unsquashed + log_det).mean(dim=-1)
    return entropy


def evaluate_agent(agent, args):
    env = gym.make(args.env)

    agent.eval()

    cum_rew = 0.0
    obs = env.reset()[0]
    for t in count():
        with torch.no_grad():
            obs_ = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            obs_ = obs_.cuda()

            action = agent.actor(obs_)[1]
            action = action.view(-1).to("cpu").numpy()

        next_obs, rew, done = env.step(action)[:3]
        cum_rew += rew
        
        terminate = done or ((t + 1) == args.max_steps)

        if terminate:
            break
        else:
            obs = next_obs

    return cum_rew
