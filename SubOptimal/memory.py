from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ("obs", "action", "next_obs", "rew", "done"))


def convert(batch):
    obs, action, next_obs, rew, done = (
        batch.obs,
        batch.action,
        batch.next_obs,
        batch.rew,
        batch.done,
    )
    obs = torch.tensor(np.array(obs), dtype=torch.float32).cuda()
    action = torch.tensor(np.array(action), dtype=torch.float32).cuda()
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).cuda()
    rew = torch.tensor(np.array(rew), dtype=torch.float32).cuda()
    done = torch.tensor(np.array(done), dtype=torch.float32).cuda()

    return obs, action, next_obs, rew, done


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

    def take_recent(self, batch_size):
        instances = [self.memory[idx] for idx in range(-batch_size, 0)]
        batch = Transition(*zip(*instances))
        return convert(batch)

    def __len__(self):
        return len(self.memory)
