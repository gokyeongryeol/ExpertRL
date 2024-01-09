import os
import h5py
import numpy as np
import gym

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from algo import Dynamics


from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ("obs", "action", "next_obs"))


def convert(batch):
    obs, action, next_obs = (
        batch.obs,
        batch.action,
        batch.next_obs,
    )
    obs = torch.tensor(np.array(obs), dtype=torch.float32).cuda()
    action = torch.tensor(np.array(action), dtype=torch.float32).cuda()
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).cuda()

    return obs, action, next_obs


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


memory = Memory(int(3e6))

for name in ["expert", "medium", "random"]:
    with h5py.File(f"../demon/walker2d_{name}-v2.hdf5", "r") as f:
        obs, action, next_obs = f["observations"], f["actions"], f["next_observations"]
        obs, action, next_obs = np.array(obs), np.array(action), np.array(next_obs)
        for o, a, no in zip(obs, action, next_obs):
            memory.push(o, a, no)

    f.close()

os.makedirs("ckpt", exist_ok=True)
os.makedirs("metric", exist_ok=True)
   
env = gym.make("Walker2d-v2")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

metric = {"loss": []}

dynamics = Dynamics(obs_dim, 128, act_dim, num_layer=4)
dynamics.cuda()

optimizer = optim.Adam(dynamics.parameters(), lr=1e-2)
scheduler = MultiStepLR(
            optimizer,
            milestones=[50, 80],
            gamma=0.1,
        )

for epoch in range(int(1e+2)):
    for t in range(int(1e+3)):
        dynamics.train()

        batch = memory.sample(128)
        obs, action, next_obs = batch
        loss = F.mse_loss(dynamics(obs, action), next_obs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric["loss"].append(loss.item())

    scheduler.step()

    torch.save(dynamics.state_dict(), "ckpt/Walker2d-v2_dynamics.pt")
    torch.save(metric, "metric/Walker2d-v2_dynamics.pt")
    print("Epoch: {} \t Loss: {}".format(epoch, loss.item()))

