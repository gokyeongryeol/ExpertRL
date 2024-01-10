### Experiment

#### Dynamics model (step1)
```bash
python expert.py
```

#### RL Agent (step2 & step3)

The following scripts require a huge amount of RAM memory (minimum 119GB). This is essential to retain the complete offline demonstrations within the memory buffer from which the mini-batch is selected.

There are two alternative methods:
- The most direct approach involves increasing the n_jump, initially set to 5. Although there hasn't been an ablation study determining the optimal value, setting it too high could harm training by reducing sample diversity.
- Another option is to individually sample each batch element by (i) loading one of the offline trajectories in ../64px and (ii) choosing a fixed length interval. Use the code block below as an alternative to Memory.sample(). While this significantly reduces memory usage, we discovered it to be excessively time-consuming.
```python
FILES = os.listdir("../64px")

def sample(self, batch_size, seq_len=None):
    indexes = np.random.choice(
        np.arange(len(FILES)), batch_size, replace=True
    )
    times = np.random.choice(
        np.arange(500-seq_len-1), batch_size, replace=True
    )

    instances = []
    for i in range(len(FILES)):
        if i not in indexes:
            continue

        data = np.load(f"../64px/{FILES[i]}")

        for t in times[indexes == i]:
            aux_obs = np.transpose(data['image'][t:t+seq_len+1], (0,3,1,2))
            action = data['action'][t+1:t+seq_len+1]
            rew = data['reward'][t+1:t+seq_len+1]
            done = data['is_last'][t+1:t+seq_len+1]

            instances.append((aux_obs, action, rew, done))

    batch = Transition(*zip(*instances))
    return convert(batch)
```

We would be happy if you suggest a better way in terms of memory usage or training time. Please feel free to use the issue tab or contact me by email: [kyeongryeol.go@gmail.com](kyeongryeol.go@gmail.com).

##### Non-informative
```bash
MUJOCO_GL=egl python main.py --is_e2e --alg slac --use_offline --from_scratch --seed 3061
```

##### Informative
```bash
MUJOCO_GL=egl python main.py --alg slac --use_offline --from_scratch --seed 3061
```
