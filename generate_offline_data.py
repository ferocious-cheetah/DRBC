import gym
import argparse
import os
import numpy as np
from stable_baselines3 import DQN, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import DATA, get_action_type, get_action_spec, get_state_spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment name
    parser.add_argument('--env', default='Hopper-v3', type=str)
    parser.add_argument('--max_size', default=100000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--verbatim', default=False, action='store_true')
    
    args = parser.parse_args()
    
    # Make dir
    if not os.path.exists("./offline_data"):
        os.makedirs("./offline_data")
    
    # Setup environment
    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env])
    env.seed(args.seed)
    env_action_type = get_action_type(env)
    action_dim, min_action, max_action = get_action_spec(env_action_type, env)
    state_dim, max_state = get_state_spec(env)
    
    # Load policy and prepare save path
    if 'FrozenLake-v1' in args.env:
        save_path = f'offline_data/vi-frozenlake-expert'
        policy = np.load('./experts/vi-frozenlake-expert.npy')
    elif 'FrozenLake8x8-v1' in args.env:
        save_path = f'offline_data/vi-frozenlake8x8-expert'
        policy = np.load('./experts/vi-frozenlake8x8-expert.npy')
    elif 'FrozenLakePerturbed-v1' in args.env:
        save_path = f'offline_data/vi-frozenlakeperturbed-expert'
        policy = np.load('./experts/vi-frozenlakeperturbed-expert.npy') 
    elif 'FrozenLakePerturbed8x8-v1' in args.env:
        save_path = f'offline_data/vi-frozenlakeperturbed8x8-expert'
        policy = np.load('./experts/vi-frozenlakeperturbed8x8-expert.npy')
    elif 'Hopper-v3' in args.env:
        save_path = f'offline_data/td3-hopper-expert'
        policy = TD3.load('./experts/td3-hopper-expert.zip')
    elif 'HalfCheetah-v3' in args.env:
        save_path = f'offline_data/td3-halfcheetah-expert'
        policy = TD3.load('./experts/td3-halfcheetah-expert.zip')
    elif 'Walker2d-v3' in args.env:
        save_path = f'offline_data/td3-walker2d-expert'
        policy = TD3.load('./experts/td3-walker2d-expert.zip')
    elif 'Ant-v3' in args.env:
        save_path = f'offline_data/td3-ant-expert'
        policy = TD3.load('./experts/td3-ant-expert.zip') 
    else:
        save_path = None
        policy = None

    # Collect trajectories
    states = []
    actions = []
    next_states = []
    rewards = []
    not_dones = []
    counter = 0
    while counter < args.max_size:
        state, done = env.reset(), False
        while not done:
            if 'FrozenLake' in args.env:
                action = policy[state]
            else:
                action, _ = policy.predict(state, deterministic=True)
            # Record s,a
            if 'FrozenLake' in args.env:
                states.append(state)
                actions.append(action)
            else:
                states.append(state.squeeze())
                actions.append(action.squeeze())
                
            if args.verbatim and counter % 1000 == 0:
                print(f'{counter}-th transition; (s,a): ({state}, {action})')
                
            # Step
            state, reward, done, _ = env.step(action)
            # Record s', r, not_done
            if 'FrozenLake' in args.env:
                next_states.append(state)
                rewards.append(reward)
                not_dones.append(1.0 - done)
            else:
                next_states.append(state.squeeze())
                rewards.append(reward)
                not_dones.append(1.0 - done)
            counter += 1
        if args.verbatim:
            print('=========== Trajectory Done ===========')
            if reward == 1:
                print('===========    Success !    ===========')
    print(f'Generated {len(states)} number of transitions. Truncating to args.max_size={args.max_size}.')        

    
    # Prepare to save
    replay_buffer = DATA(state_dim, action_dim, 'cpu', max_size=args.max_size)
    replay_buffer.state = np.array(states)[:args.max_size]
    replay_buffer.action = np.array(actions)[:args.max_size]
    replay_buffer.next_state = np.array(next_states)[:args.max_size]
    replay_buffer.reward = np.array(rewards)[:args.max_size]
    replay_buffer.not_done = np.array(not_dones)[:args.max_size]
    replay_buffer.size = args.max_size
    replay_buffer.ptr = replay_buffer.size
    replay_buffer.save(save_path)
    