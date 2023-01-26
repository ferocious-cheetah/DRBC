import os
import pyrallis
import gym

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random

from dataclasses import dataclass, field
from scipy.optimize import minimize
from utils import DATA, get_action_type, get_action_spec, get_state_spec
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List
from itertools import product


@dataclass
class TrainConfig:
    # Experiment
    device: str = 'cpu'
    env_name: str = 'Hopper-v3'  # OpenAI gym environment name
    seed: int = 0
    eval_seed: int = 42
    eval_freq: int = 5000
    eval_episodes: int = 10
    max_timesteps: int = 1000000
    expert_datasize: int = 200
    save_model: bool = False
    
    # DROBC
    rho: float = 0.1
    batch_size: int = 256
    actor_lr: float = 3e-4
    dual_update_freq: int = 1
    lr_decay: bool = False
    lr_decay_freq: int = 10000
    lr_decay_gamma: float = 0.8
        
    # Paths
    data_path: str = None
    save_path: str = None
    log_path: str = None
        
    # Misc
    timestamp: str = None
    note: str = None
        
    def __post_init__(self):
        self.timestamp = str(time.time())
        if self.data_path is None:
            # Infer from env_name
            temp = self.env_name.split('-')[0].lower()
            self.data_path = f'./offline_data/td3-{temp}-expert'
        if self.save_path is None:
            self.save_path = f"./runs/drobc-{self.env_name.split('-')[0]}-{self.timestamp[:10]}".lower()
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        if self.log_path is None:
            self.log_path = f"./tensorboard_logs/runs/drobc-{self.env_name.split('-')[0]}-{self.timestamp[:10]}".lower()
        
        
        
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, device: str,
                 min_action: np.ndarray, max_action: np.ndarray):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.min_action = torch.tensor(min_action, dtype=torch.float, 
                                       device=device)
        self.max_action = torch.tensor(max_action, dtype=torch.float, 
                                       device=device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device='cpu'):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = torch.clamp(self(state), self.min_action, self.max_action)
        return action.cpu().data.numpy().flatten()
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(load_path, 
                                        map_location=torch.device(device)))
        

class Discrete_Actor(nn.Module):
    def __init__(self, state_dim: int, num_of_action: int, device: str,
                 min_action: np.ndarray, max_action: np.ndarray):
        super(Discrete_Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_of_action)
        self.out = nn.LogSoftmax(dim=1)
        
        self.min_action = torch.tensor(min_action, dtype=torch.float, 
                                       device=device)
        self.max_action = torch.tensor(max_action, dtype=torch.float, 
                                       device=device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        log_prob = self.out(self.l3(x))
        return log_prob
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device='cpu'):
        state = torch.tensor(state.reshape(1,-1), device=device, dtype=torch.float32)
        log_prob = self(state)
        return log_prob.argmax(dim=1).cpu().detach().numpy().flatten()
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(load_path, 
                                        map_location=torch.device(device)))
        
class DROBC:
    def __init__(self, state_dim, action_dim, min_action, max_action, device,
                 env_action_type, rho=0.1, actor_lr=6e-4, lr_decay=False, lr_decay_freq=10000,
                 lr_decay_gamma=0.8):
        # Important - need to supply tensors to clamp actions at:
        # 1. Actor.act
        # 2. model predict action
        self.min_action = min_action
        self.max_action = max_action
        
        # Learning rates
        self.actor_lr = actor_lr
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay = lr_decay
        
        # Other class/robust variables
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.env_action_type = env_action_type
        self.rho = rho
        
        # Initialize
        if env_action_type == 'discrete':
            self.actor = Discrete_Actor(state_dim, max_action+1, device, min_action, max_action).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, device, min_action, max_action).to(device)
        # Optimizer and lr scheduler
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        if lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer,
                                                                       gamma=lr_decay_gamma)
        
        self.min_dual = -1.0
        self.max_dual = (1 + rho) * (np.linalg.norm(max_action-min_action) ** 2)
        
        # Keep record of data mean/std
        # This will be recorded after the first time call to self.train()
        self.mean = None
        self.std = None
        
        # Misc.
        self.zero = torch.tensor(0, device=device)
    
    def optimize_dual(self, pi_theta, pi_e):
        sup = np.max(np.linalg.norm(pi_theta - pi_e, axis=1)**2)  # Calculate norm across columns
        def dual_func(eta):
            '''Total variation uncertainty dual function.'''
            res = np.mean(np.maximum(np.linalg.norm(pi_theta-pi_e, axis=1)**2 - eta, 0))
            res += self.rho * np.maximum(sup-eta, 0)
            res += eta
            return res
        bounds = [(self.min_dual, self.max_dual)]
        res = minimize(dual_func, x0=0, method='SLSQP', bounds=bounds, tol=1e-3)
        if not res.success:
            raise ValueError("Dual optimization failed!")
        return res.fun, sup
    
    def optimize_dual_discrete(self, nll_loss):
        sup = np.max(nll_loss)
        def dual_func(eta):
            '''Total variation uncertainty dual function.'''
            res = np.mean(np.maximum(nll_loss - eta, 0))
            res += self.rho * np.maximum(sup-eta, 0)
            res += eta
            return res
        bounds = [(-1, (1+self.rho) * sup)]
        res = minimize(dual_func, x0=0, method='SLSQP', bounds=bounds, tol=1e-3)
        if not res.success:
            raise ValueError("Dual optimization failed!")
        return res.fun, sup
    
    def save_factor(self, data, save_path=None):
        self.mean, self.std = data.get_factor(tensor=False)
        if save_path is not None:
            np.save(f'{save_path}/mean', self.mean)
            np.save(f'{save_path}/std', self.std)   
        
    def train(self, data, trn_steps, dual_update_freq, batch_size=100, writer=None, writer_base=0):
        for i in range(trn_steps):
            _, action, _, _, _, normalized_state = data.sample(batch_size) 
            if i % dual_update_freq == 0:  # Update dual variable (including i=0)
                if self.env_action_type == 'discrete':
                    nll_loss = F.nll_loss(self.actor(normalized_state), action.long().view(-1),
                                          reduction='none').detach().cpu().numpy()
                    dual, sup = self.optimize_dual_discrete(nll_loss)
                    dual, sup = torch.tensor(dual, device=self.device), torch.tensor(sup, device=self.device)
                    
                else:
                    # Inquire actor pi_theta(s)
                    # Detach to get values ONLY to optimize dual variable
                    pi_theta = self.actor(normalized_state).detach().cpu().numpy()
                    pi_e = action.cpu().numpy()
                    # Optimize dual variable
                    dual, sup = self.optimize_dual(pi_theta, pi_e)
                    if sup is not None:
                        sup = torch.tensor(sup, device=self.device)
                    dual = torch.tensor(dual, device=self.device)
            if self.env_action_type == 'discrete':
                # Calculate DRO loss with gradient info
                dro_loss = F.nll_loss(self.actor(normalized_state), action.long().view(-1), reduction='none')
                dro_loss -= dual
                dro_loss = torch.mean(torch.maximum(dro_loss, self.zero))
                dro_loss += self.rho * torch.maximum(sup - dual, self.zero)
                dro_loss += dual
            else:
                # Calculate DRO loss with gradient info
                dro_loss = torch.square(torch.linalg.norm(self.actor(normalized_state) - action, axis=1)) - dual
                dro_loss = torch.mean(torch.maximum(dro_loss, self.zero))
                dro_loss += self.rho * torch.maximum(sup - dual, self.zero)
                dro_loss += dual
            
            # Actor gradient update
            self.actor_optimizer.zero_grad()
            dro_loss.backward()
            self.actor_optimizer.step()
            
            # Scheduler step
            if self.lr_decay and i % self.lr_decay_freq == 0 and writer_base != 0:
                self.lr_scheduler.step()
                if writer is not None:
                    lrs = self.lr_scheduler.get_last_lr()
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'Training/learning_rate_{i}', lrs[i], i+writer_base)
            
            # Log tensorboard
            if writer is not None:
                writer.add_scalar('Training/optimal_dual', dual, i+writer_base)
                writer.add_scalar('Loss/dro_loss', dro_loss, i+writer_base)
                if sup is not None:
                    writer.add_scalar('Training/estimated_supremum_policy_gap', sup, i+writer_base)
        if writer is not None:
            writer.flush()
    
    def predict(self, state):
        # Standardized state
        normalized_state = (state - self.mean) / self.std
        action = self.actor.act(normalized_state, self.device)
        if self.env_action_type == 'discrete':
            action = action.item()
        elif self.env_action_type == 'continuous':
            action = action[0]
        elif self.env_action_type == 'multi_continuous':
            pass
        else:
            raise NotImplementedError
        return action
    
    def load(self, load_path):
        self.actor.load(load_path, device=self.device)
        
    
def eval_policy(policy, env_name, seed, eval_episodes=10, with_gauss=False, action_dim=0, var=0):
    env = gym.make(env_name)
    rewards = []
    eps_reward = 0
    for i in range(eval_episodes):
        if 'FrozenLake' in env_name:
            state, done = env.reset(), False
            env.seed(seed+i)
        else:
            state, done = env.reset(seed=seed+i), False
        while not done:
            action = policy.predict(state)
            if with_gauss:
                action += np.random.multivariate_normal(np.zeros(action_dim), var*np.identity(action_dim))
            state, reward, done, _ = env.step(action)
            eps_reward += reward
        rewards.append(eps_reward)
        eps_reward = 0
    return np.mean(rewards), np.std(rewards)


@pyrallis.wrap()
def main(cfg: TrainConfig):
    env = gym.make(cfg.env_name)
    # Set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Get specs 
    env_action_type = get_action_type(env)
    action_dim, min_action, max_action = get_action_spec(env_action_type, env)
    state_dim, max_state = get_state_spec(env)
    
    # Stamp the run
    pyrallis.dump(cfg, open(f'{cfg.save_path}/config.yaml', 'w'))

    # Initialize
    data = DATA(state_dim, action_dim, cfg.device, max_size=cfg.expert_datasize)
    data.load(cfg.data_path)
    writer = SummaryWriter(cfg.log_path)
    policy = DROBC(state_dim, action_dim, min_action, max_action, cfg.device, env_action_type, cfg.rho,
                    cfg.actor_lr, cfg.lr_decay, cfg.lr_decay_freq, cfg.lr_decay_gamma)
    # Save factor
    policy.save_factor(data, cfg.save_path)

    # Train!
    trn_iters = 0
    max_reward = -np.inf
    while trn_iters < cfg.max_timesteps:
        policy.train(data, trn_steps=cfg.eval_freq, dual_update_freq=cfg.dual_update_freq, 
                    batch_size=cfg.batch_size, writer=writer, writer_base=trn_iters)
        avg, std = eval_policy(policy, cfg.env_name, cfg.eval_seed, cfg.eval_episodes)

        # Record scores
        writer.add_scalar('Evaluation/episode_reward_avg', avg, trn_iters+cfg.eval_freq)
        writer.add_scalar('Evaluation/episode_reward_std', std, trn_iters+cfg.eval_freq)
                    
        if cfg.save_model:          
            if avg > max_reward:  # Check if outperforms
                policy.actor.save(f'{cfg.save_path}/best_model.pt')
                max_reward = avg
            # Save model
            policy.actor.save(f'{cfg.save_path}/model_{trn_iters+cfg.eval_freq}.pt')
        
        # Move forward while loop
        trn_iters += cfg.eval_freq

    # Clean up
    writer.close()
    
if __name__ == "__main__":
    main()