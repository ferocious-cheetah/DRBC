import pyrallis
import gym
import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from drobc import DROBC
from bc import BC
from utils import get_action_type, get_action_spec, get_state_spec
from utils import save_dict
from typing import Optional, List


@dataclass
class EvalConfig:
    # Evaluation
    env_name: str = 'Hopper-v3'
    seed: int = 0
    eval_episodes: int = 10
    drobc: bool = True
    
    # Paths
    model_name: str = None
    load_path: str = None
    save_path: str = None


def eval_on_hopper(policy, seed, action_dim, eval_episodes=10):
    '''
    settings = ['gravity', 'thigh_joint_stiffness', 'leg_joint_stiffness',
                'foot_joint_stiffness', 'actuator_ctrlrange', 'joint_damping_p',
                'joint_frictionloss', 'action']
    '''
    out = {}
    env = gym.make('HopperPerturbed-v3')
    springref = 0.2
    
    # 'gravity': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    means, stds = [], []
    xs = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    xaxis = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, gravity=env.gravity*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['gravity'] = [means, stds, xaxis]
    
    # 'thigh_joint_stiffness': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    means, stds = [], []
    xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    springref=springref,
                                    thigh_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['thigh_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'leg_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    springref=springref,
                                    leg_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))   
    out['leg_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'foot_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    springref=springref,
                                    foot_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards)) 
    out['foot_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    actuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['joint_damping_p'] = [means, stds, xaxis]
    
    
    # 'joint_frictionloss': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    means, stds = [], []
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    joint_frictionloss=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['joint_frictionloss'] = [means, stds, xaxis]
    
    # 'action': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    means, stds = [], []
    xs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i), False
            while not done:
                action = policy.predict(state)
                action += np.random.multivariate_normal(np.zeros(action_dim), x*np.identity(action_dim))
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['action'] = [means, stds, xaxis]
    return out


def eval_on_halfcheetah(policy, seed, action_dim, eval_episodes=10):
    '''
    settings = ['gravity', 'back_joint_stiffness', 'front_joint_stiffness',
                'front_actuator_ctrlrange', 'back_actuator_ctrlrange', 'joint_damping_p',
                'joint_frictionloss', 'action']
    '''
    out = {}
    env = gym.make('HalfCheetahPerturbed-v3')
    
    # 'gravity': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    means, stds = [], []
    xs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    xaxis = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, gravity=env.gravity*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['gravity'] = [means, stds, xaxis]
    
    # 'back_joint_stiffness': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    means, stds = [], []
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    xaxis = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    bthigh_joint_stiffness=env.bthigh_joint_stiffness*(1+x),
                                    bshin_joint_stiffness=env.bshin_joint_stiffness*(1+x),
                                    bfoot_joint_stiffness=env.bfoot_joint_stiffness*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['back_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'front_joint_stiffness': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    means, stds = [], []
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    xaxis = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    fthigh_joint_stiffness=env.fthigh_joint_stiffness*(1+x),
                                    fshin_joint_stiffness=env.fshin_joint_stiffness*(1+x),
                                    ffoot_joint_stiffness=env.ffoot_joint_stiffness*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))   
    out['front_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'front_actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    factuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards)) 
    out['front_actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'back_actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    bactuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['back_actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    xaxis = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['joint_damping_p'] = [means, stds, xaxis]
    
    
    # 'joint_frictionloss': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    means, stds = [], []
    xs = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    joint_frictionloss=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['joint_frictionloss'] = [means, stds, xaxis]
    
    # 'action': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    means, stds = [], []
    xs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
#     xs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            np.random.seed(seed+i)
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i), False
            while not done:
                action = policy.predict(state)
                action += np.random.multivariate_normal(np.zeros(action_dim), x*np.identity(action_dim))
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['action'] = [means, stds, xaxis]
    return out

def eval_on_walker2d(policy, seed, action_dim, eval_episodes=10):
    '''
    settings = ['gravity', 'thigh_joint_stiffness', 'leg_joint_stiffness',
                'foot_joint_stiffness', 'actuator_ctrlrange', 'thigh_joint_damping_p',
                'leg_joint_damping_p', 'foot_joint_damping_p', 'joint_frictionloss', 'action']
    '''
    out = {}
    env = gym.make('Walker2dPerturbed-v3')
    
    # 'gravity': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    means, stds = [], []
    xs = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    xaxis = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, gravity=env.gravity*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['gravity'] = [means, stds, xaxis]
    
    # 'thigh_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    thigh_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['thigh_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'leg_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    leg_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))   
    out['leg_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'foot_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    foot_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards)) 
    out['foot_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    actuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'thigh_joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    thigh_joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['thigh_joint_damping_p'] = [means, stds, xaxis]
    
    # 'leg_joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    leg_joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['leg_joint_damping_p'] = [means, stds, xaxis]
    
    # 'foot_joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    foot_joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['foot_joint_damping_p'] = [means, stds, xaxis]
    
    
    # 'joint_frictionloss': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    means, stds = [], []
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    joint_frictionloss=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['joint_frictionloss'] = [means, stds, xaxis]
    
    # 'action': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    means, stds = [], []
    xs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i), False
            while not done:
                action = policy.predict(state)
                action += np.random.multivariate_normal(np.zeros(action_dim), x*np.identity(action_dim))
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['action'] = [means, stds, xaxis]
    return out

def eval_on_ant(policy, seed, action_dim, eval_episodes=10):
    '''
    settings = ['gravity', 'hip_joint_stiffness', 'ankle_joint_stiffness',
                'hip_actuator_ctrlrange', 'ankle_actuator_ctrlrange', 'hip_joint_damping_p',
                'ankle_joint_damping_p', 'hip_joint_frictionloss', 'ankle_joint_frictionloss', 'action']
    '''
    out = {}
    env = gym.make('AntPerturbed-v3')
    
    # 'gravity': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    means, stds = [], []
    xs = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    xaxis = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, gravity=env.gravity*(1+x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['gravity'] = [means, stds, xaxis]
    
    # 'hip_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    hip_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['hip_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'ankle_joint_stiffness': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means, stds = [], []
    xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    ankle_joint_stiffness=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))   
    out['ankle_joint_stiffness'] = [means, stds, xaxis]
    
    
    # 'hip_actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    hip_actuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['hip_actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'ankle_actuator_ctrlrange': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    means, stds = [], []
    xs = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i,
                                    ankle_actuator_ctrlrange=(-x,x)), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['ankle_actuator_ctrlrange'] = [means, stds, xaxis]
    
    
    # 'hip_joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    hip_joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['hip_joint_damping_p'] = [means, stds, xaxis]
    
    # 'ankle_joint_damping_p': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    means, stds = [], []
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    xaxis = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    ankle_joint_damping_p=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['ankle_joint_damping_p'] = [means, stds, xaxis]
    
    # 'hip_joint_frictionloss': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    means, stds = [], []
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    hip_joint_frictionloss=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['hip_joint_frictionloss'] = [means, stds, xaxis]
    
    
    # 'ankle_joint_frictionloss': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    means, stds = [], []
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i, 
                                    ankle_joint_frictionloss=x), False
            while not done:
                action = policy.predict(state)
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['ankle_joint_frictionloss'] = [means, stds, xaxis]
    
    # 'action': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    means, stds = [], []
    xs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    xaxis = xs
    for x in xs:
        rewards = []
        for i in range(eval_episodes):
            eps_reward = 0.0
            state, done = env.reset(seed=seed+i), False
            while not done:
                action = policy.predict(state)
                action += np.random.multivariate_normal(np.zeros(action_dim), x*np.identity(action_dim))
                state, reward, done, _ = env.step(action)
                eps_reward += reward
            rewards.append(eps_reward)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    out['action'] = [means, stds, xaxis]
    return out
        
@pyrallis.wrap()
def main(cfg: EvalConfig):
    env = gym.make(cfg.env_name)
    # Set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Get specs
    env_action_type = get_action_type(env)
    action_dim, min_action, max_action = get_action_spec(env_action_type, env)
    state_dim, max_state = get_state_spec(env)

    # Load model
    if cfg.drobc:
        policy = DROBC(state_dim, action_dim, min_action, max_action, 'cpu', env_action_type)
        policy.load(f'{cfg.load_path}/{cfg.model_name}')
        policy.mean = np.load(f'{cfg.load_path}/mean.npy')
        policy.std = np.load(f'{cfg.load_path}/std.npy')
    else:
        policy = BC(state_dim, action_dim, min_action, max_action, 'cpu', env_action_type)
        policy.load(f'{cfg.load_path}/{cfg.model_name}')
        policy.mean = np.load(f'{cfg.load_path}/mean.npy')
        policy.std = np.load(f'{cfg.load_path}/std.npy')
    # Evaluation
    if 'Hopper' in cfg.env_name:
        out = eval_on_hopper(policy, cfg.seed, action_dim, eval_episodes=cfg.eval_episodes)
    elif 'HalfCheetah' in cfg.env_name:
        out = eval_on_halfcheetah(policy, cfg.seed, action_dim, eval_episodes=cfg.eval_episodes)
    elif 'Walker2d' in cfg.env_name:
        out = eval_on_walker2d(policy, cfg.seed, action_dim, eval_episodes=cfg.eval_episodes)
    elif 'Ant' in cfg.env_name:
        out = eval_on_ant(policy, cfg.seed, action_dim, eval_episodes=cfg.eval_episodes)
    else:
        raise NotImplementedError
               
    # Save results
    save_dict(save_path=cfg.save_path, data=out)
    
if __name__ == "__main__":
    main()