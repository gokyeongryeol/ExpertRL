import h5py
import numpy as np

from algo import AWAC, SAC, TD_Ratio
from memory import Memory

Q_LAYER = {"AWAC": 2, "SAC": 2, "TD_Ratio": 2}
P_LAYER = {"AWAC": 4, "SAC": 2, "TD_Ratio": 2}
MIN_LOG = {"AWAC": -6, "SAC": -20, "TD_Ratio": -20}
MAX_LOG = {"AWAC": 0, "SAC": 2, "TD_Ratio": 2}
TUNABLE = {"AWAC": False, "SAC": True, "TD_Ratio": True}
ALPHA = {"AWAC": 0.0, "SAC": 1.0, "TD_Ratio": 1.0}
Q_LR = {"AWAC": 3e-4, "SAC": 3e-4, "TD_Ratio": 3e-4}
P_LR = {"AWAC": 3e-4, "SAC": 3e-4, "TD_Ratio": 3e-4}
P_WD = {"AWAC": 1e-4, "SAC": 0.0, "TD_Ratio": 0.0}
MEMORY_SIZE = {"AWAC": 1e6, "SAC": 1e6, "TD_Ratio": 1e6}


def make_offline_agent(obs_dim, act_dim, act_limit, args):
    off_agent = globals()[args.offline_alg](
        obs_dim,
        args.hid_dim,
        act_dim,
        Q_LAYER[args.offline_alg],
        P_LAYER[args.offline_alg],
        act_limit,
        MIN_LOG[args.offline_alg],
        MAX_LOG[args.offline_alg],
        TUNABLE[args.offline_alg],
        ALPHA[args.offline_alg],
        args.gamma,
        args.rho,
        q_lr=Q_LR[args.offline_alg],
        p_lr=P_LR[args.offline_alg],
        p_wd=P_WD[args.offline_alg],
    )
    off_agent = off_agent.cuda()

    off_memory = Memory(int(MEMORY_SIZE[args.offline_alg]))
    with h5py.File("../demon/walker2d_medium-v2.hdf5", "r") as f:
        obs, action, next_obs = f["observations"], f["actions"], f["next_observations"]
        rew, done = f["rewards"], f["terminals"]
        obs, action, next_obs = np.array(obs), np.array(action), np.array(next_obs)
        rew, done = np.array(rew), np.array(done)

        for o, a, no, r, d in zip(obs, action, next_obs, rew, done):
            off_memory.push(o, a, no, r, d)
    f.close()

    return off_agent, off_memory


def make_online_agent(obs_dim, act_dim, act_limit, args):
    on_agent = globals()[args.online_alg](
        obs_dim,
        args.hid_dim,
        act_dim,
        Q_LAYER[args.online_alg],
        P_LAYER[args.online_alg],
        act_limit,
        MIN_LOG[args.online_alg],
        MAX_LOG[args.online_alg],
        TUNABLE[args.online_alg],
        ALPHA[args.online_alg],
        args.gamma,
        args.rho,
        q_lr=Q_LR[args.online_alg],
        p_lr=P_LR[args.online_alg],
        p_wd=P_WD[args.online_alg],
    )
    on_agent = on_agent.cuda()

    if "TD_Ratio" in args.online_alg:
        on_agent.set_beta(args.beta)

    on_memory = Memory(int(MEMORY_SIZE[args.online_alg]))

    return on_agent, on_memory
