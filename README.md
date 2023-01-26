# Provably Improving Behavioral Cloning with Distributionally Robust Optimization
This is an anonymous code submission for ICML'23.

Our method is tested on three OpenAI Gym MuJoCo continuous-action control tasks: `Hopper-v3`, `HalfCheetah-v3`, and `Walker2d-v3`. **Thus it is required that MuJoCo is properly installed prior to using this repo**.

## Setup
Install requirements:
```
pip install -r requirements.txt
```
One note on the version of `gym`. Since we use pre-trained agents provided in [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), the package `stable-baselines3` is required. However, it uses `gym<=0.21.0` which may have some seeding issues. We recommend you first install `stable-baselines3` and then force install `gym==0.22.0`.

Next, you need to properly register the perturbed Gym environments which are placed under the folder *envs*. A recommended way to do this: first, go the gym folder which you installed in your python *site-packages*; second, place *hopper_perturbed.py*, *half_cheetah_perturbed.py*, and *walker2d_perturbed.py* under *gym/envs/mujoco*; now, add the following to *\__init__.py* under *gym/envs*:
```
register(
    id="HopperPerturbed-v3",
    entry_point="gym.envs.mujoco.hopper_perturbed:HopperPerturbedEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="HalfCheetahPerturbed-v3",
    entry_point="gym.envs.mujoco.half_cheetah_perturbed:HalfCheetahPerturbedEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id="Walker2dPerturbed-v3",
    max_episode_steps=1000,
    entry_point="gym.envs.mujoco.walker2d_perturbed:Walker2dPerturbedEnv",
)
```
You can test this by running:
```
import gym

gym.make('HopperPerturbed-v3')
```

## Instructions
Here we use `Hopper-v3` as an example. Also make sure that you are under the *DRBC* folder when executing anything in this section. 

### Data Generation
To generate expert demonstration data specifically for `Hopper-v3`, please do
```
python generate_offline_data.py --env='Hopper-v3' --max_size=100000 --seed=42 --verbatim
```
This will generate expert data of size 100000 using the pre-trained TD3 rollouts and conform it to be readily used by our agents. `--verbatim` randomly picks transitions to print to terminal.

### Train policies
This implementation uses `pyrallis` to handle argument parsing. We have provided the configuration files in the folder *configs* needed to reproduce our results. To train BC and DR-BC, please do
```
python bc.py --config_path=configs/hopper/bc_hopper.yaml
python drobc.py --config_path = configs/hopper/drobc_hopper.yaml
```

### Evaluate policies
Before evaluating the trained agents, you need to update the files *configs/hopper/eval_bc_{env}.yaml* and *configs/hopper/eval_bc_{env}.yaml*, and *env* should be among \{'hopper', 'halfcheetah', 'walker2d'\}. Please go the the *runs* folder to record the filename of your last runs. For example, `bc-hopper-1674234241` means it's BC trained on `Hopper-v3` when the time is `1674234241`. Using this as an example, then you need to update two things in the YAML file:

1. `load_path`: runs/bc-hopper-1674234241
2. `save_path`: whereever_you_want_it_to_be.json

Now please do
```
python eval_policy.py --config_path=configs/hopper/eval_bc_hopper.yaml
```
The output file will be in the format of JSON. Please see `eval_policy.py` for the details of evaluation on the perturbed environments.