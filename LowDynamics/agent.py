import os
import torch
import numpy as np
from collections import deque, namedtuple
from algo import Dynamics, SLAC, SLVM


Transition = namedtuple("Transition", ("aux_obs", "action", "rew", "done"))


def convert(batch):
    aux_obs_seq, action_seq, rew_seq, done_seq = (
        batch.aux_obs,
        batch.action,
        batch.rew,
        batch.done,
    )
    aux_obs_seq = torch.tensor(np.array(aux_obs_seq), dtype=torch.float32).div_(255.0).cuda()
    action_seq = torch.tensor(np.array(action_seq), dtype=torch.float32).cuda()
    rew_seq = torch.tensor(np.array(rew_seq), dtype=torch.float32).cuda()
    done_seq = torch.tensor(np.array(done_seq), dtype=torch.float32).cuda()

    return aux_obs_seq, action_seq, rew_seq, done_seq


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        indexes = np.random.choice(
            np.arange(len(self.memory)), batch_size, replace=False
        )
        instances = [self.memory[idx] for idx in indexes]
        batch = Transition(*zip(*instances))
        return convert(batch)

    def __len__(self):
        return len(self.memory)


def load_dynamics(obs_dim, hid_dim, act_dim, args):
    dynamics = Dynamics(obs_dim, hid_dim, act_dim, num_layer=4)
    dynamics.cuda()
    dynamics.load_state_dict(torch.load(f"ckpt/{args.env}_dynamics.pt"))
    dynamics.eval()

    for param in dynamics.parameters():
        param.requires_grad_(False)

    return dynamics


def make_offline_agent(obs_dim, act_dim, act_limit, args):
    dynamics = None if args.is_e2e else load_dynamics(obs_dim, 128, act_dim, args)
    slvm = SLVM(256, 256, obs_dim, 256, act_dim, dynamics=dynamics)

    agent = SLAC(
        slvm,
        obs_dim,
        act_dim,
        act_limit,
        args.seq_len,
        p_layer=4,
        min_log=-6,
        max_log=0,
        tunable=False,
        alpha=0.0,
        p_wd=1e-4,
    )
    agent = agent.cuda()

    memory = Memory(args.memory_size)

    for file in os.listdir("../64px"):
        data = np.load(f"../64px/{file}")
        for t in range(0, 501-args.seq_len, args.n_jump):
            aux_obs = np.transpose(data['image'][t:t+args.seq_len+1], (0,3,1,2))
            action = data['action'][t+1:t+args.seq_len+1]
            rew = data['reward'][t+1:t+args.seq_len+1]
            done = data['is_last'][t+1:t+args.seq_len+1]

            memory.push(aux_obs, action, rew, done)

    return agent, memory

def make_online_agent(obs_dim, act_dim, act_limit, args):
    dynamics = None if args.is_e2e else load_dynamics(obs_dim, 128, act_dim, args)
    slvm = SLVM(256, 256, obs_dim, 256, act_dim, dynamics=dynamics)

    agent = SLAC(
        slvm,
        obs_dim,
        act_dim,
        act_limit,
        args.seq_len,
        min_log=-20,
        max_log=2,
        tunable=True,
        alpha=1.0,
        p_wd=0.0,
    )
    agent = agent.cuda()

    memory = Memory(args.memory_size)

    return agent, memory
