# Provably Improving Behavioral Cloning with Distributionally Robust Optimization
This is an anonymous code submission for ICML'23.

Our method is tested on three OpenAI Gym MuJoCo continuous-action control tasks: `Hopper-v3`, `HalfCheetah-v3`, and `Walker2d-v3`. **Thus it is required that MuJoCo is properly installed prior to using this repo**.

## Setup
Install requirements:
```
pip install -r requirements.txt
```
One note on the version of `gym`. Since we use pre-trained agents provided in [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), the package `stable-baselines3` is required. However, it uses `gym<=0.21.0` which may have some seeding issues. We recommend you first install `stable-baselines3` and then force install `gym==0.22.0`.

Next, you need to properly register the perturbed Gym environments which are placed under the folder *envs*. A recommended way to do this: first, go the gym folder which you installed in your python *site-packages*; second, place hopper_perturbed.py, half_cheetah_perturbed.py, and walker2d_perturbed.py under gym/envs/mujoco; now, add the following to \__init__.py under gym/envs:
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
