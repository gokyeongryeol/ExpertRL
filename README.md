# Expert-guided Data-efficient RL

[Case 1](./SubOptimal/SCRIPTS.md): sub-optimal RL agent as expert knowledge

[Case 2](./LowDynamics/SCRIPTS.md): low-dimensional dynamics as expert knowledge

### Dataset

Offline demonstration from Walker2d-v2 environment of [Mujoco](https://github.com/openai/mujoco-py)
```bash
mkdir demon && cd demon/
wget https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5
wget https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5
wget https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_random-v2.hdf5
```
Offline demonstration from domain name 'walker' and task name 'walk' of [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
- [link](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI?usp=sharing) from [V-D4RL](https://github.com/conglu1997/v-d4rl)
- download the folder `main/walker_walk/medium/64px`
- the overall file structure becomes:
```
ExpertRL
└───64px
│   └───20220109T013713-488d0464873a40c99446c53b1468e1c9-501.npz
│   └───20220109T013713-681bef276feb456c8096851ae536d2a2-501.npz
│   │   ...
│   └───20220109T013743-e3b1c9be2e534355a7427ffcceb9d042-501.npz
└───demon
│   └───walker2d_expert-v2.hdf5
│   └───walker2d_medium-v2.hdf5
│   └───walker2d_random-v2.hdf5
└───dmc2gym
└───LowDynamics
│   └───SCRIPTS.md
│   │   ...
└───SubOptimal
│   └───SCRIPTS.md
│   │   ...
└───README.md
```

### Installation

#### docker container
```bash
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
docker run -it --shm-size=8G \
    --gpus=all --restart=always \
    pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel \
    /bin/bash
```

#### additional library
```bash
pip install h5py
```

#### simulated environment
We used the image wrapper in [dmc2gym](https://github.com/denisyarats/dmc2gym) with a few modifications using local install, which is necessary for the updated mujoco version.
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco && mv mujoco210 ~/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

pip3 install -U 'mujoco-py<2.2,>=2.1'
pip install mujoco
pip install gym[mujoco]

pip install 'cython<3'
apt install libosmesa6-dev libgl1-mesa-glx libglfw3
ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

cd dmc2gym/ && pip install -e .
```
