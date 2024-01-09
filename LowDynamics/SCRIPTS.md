### Experiment

#### Dynamics model
```bash
python expert.py
```

#### Non-informative
```bash
MUJOCO_GL=egl python main.py --is_e2e --alg slac --use_offline --from_scratch --seed 3061
```

#### Informative
```bash
MUJOCO_GL=egl python main.py --alg slac --use_offline --from_scratch --seed 3061
```
