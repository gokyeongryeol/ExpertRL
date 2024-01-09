### Experiment

#### Maximum Entropy RL(ME-RL)
```bash
python main.py --online_alg SAC --seed 3061
```

#### KL-regularized RL (KL-RL)
```bash
python main.py --use_offline --online_alg SAC --offline_alg AWAC --from_scratch --seed 3061
```

#### TD-Ratio
```bash
python main.py --use_offline --beta 0.01 --online_alg TD_Ratio --offline_alg AWAC --from_scratch --seed 3061
```
