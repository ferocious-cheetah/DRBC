import numpy as np
import torch
import gym
import warnings
import json

def setup_result_container(env_name):
    res = {}
    if 'hopper' in env_name:
        res['nominal'] = []
        res['gravity'] = [[], [], []]
        res['thigh_joint_stiffness'] = [[], []]
        res['leg_joint_stiffness'] = [[], []]
        res['foot_joint_stiffness'] = [[], []]
        res['actuator_ctrlrange'] = [[], []]
        res['joint_damping_p'] = [[], []]
        res['joint_frictionloss'] = [[], []]
        res['action'] = [[], [], []]     
    elif 'halfcheetah' in env_name:
        res['nominal'] = []
        res['gravity'] = [[], [], []]
        res['back_joint_stiffness'] = [[], []]
        res['front_joint_stiffness'] = [[], []]
        res['front_actuator_ctrlrange'] = [[], []]
        res['back_actuator_ctrlrange'] = [[], []]
        res['joint_damping_p'] = [[], []]
        res['joint_frictionloss'] = [[], []]
        res['action'] = [[], [], []]
    elif 'walker2d' in env_name:
        res['nominal'] = []
        res['gravity'] = [[], [], []]
        res['thigh_joint_stiffness'] = [[], []]
        res['leg_joint_stiffness'] = [[], []]
        res['foot_joint_stiffness'] = [[], []]
        res['actuator_ctrlrange'] = [[], []]
        res['thigh_joint_damping_p'] = [[], []]
        res['leg_joint_damping_p'] = [[], []]
        res['foot_joint_damping_p'] = [[], []]
        res['joint_frictionloss'] = [[], []]
        res['action'] = [[], [], []]
    elif 'ant' in env_name:
        res['nominal'] = []
        res['gravity'] = [[], [], []]
        res['hip_joint_stiffness'] = [[], []]
        res['ankle_joint_stiffness'] = [[], []]
        res['hip_actuator_ctrlrange'] = [[], []]
        res['ankle_actuator_ctrlrange'] = [[], []]
        res['hip_joint_damping_p'] = [[], []]
        res['ankle_joint_damping_p'] = [[], []]
        res['hip_joint_frictionloss'] = [[], []]
        res['ankle_joint_frictionloss'] = [[], []]
        res['action'] = [[], [], []]
    else:
        raise NotImplementedError
    return res   

def save_dict(save_path, data):
    with open(save_path, 'w') as f: 
        json.dump(data, f)
        
def load_dict(load_path):
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data

class DATA(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Supply standardized observation
        self.mean = None
        self.std = None
        self.normalized_state = np.zeros((max_size, state_dim))

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        if self.size < batch_size:
            warnings.warn("The size of dataset is less than batch size!")
            batch_size = self.size
        
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.normalized_state[ind]).to(self.device)
        )


    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)
        
    
    def get_factor(self, tensor=False):
        if tensor:
            return (
                torch.FloatTensor(self.mean).to(self.device),
                torch.FloatTensor(self.std).to(self.device)
            )
        else:
            return (
                self.mean,
                self.std
            )


    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        # resize data first!
        self.state.resize((self.size, self.state_dim))
        self.action.resize((self.size, self.action_dim))
        self.next_state.resize((self.size, self.state_dim))
        self.reward.resize((self.size, 1))
        self.not_done.resize((self.size, 1))
        self.normalized_state.resize((self.size, self.state_dim))
        
        # then load!
        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]
        
        # Standardize observation
        self.mean = np.mean(self.state, axis=0)
        self.std = np.std(self.state, axis=0)
        self.std[self.std==0] = 1e-6  # Important: fix the case of zero std!
        self.normalized_state[:self.size] = (self.state - self.mean) / self.std
        
            
def get_action_type(env: gym.Env):
    """
    Method to get the action type to choose prob. dist. 
    to sample actions from NN logits output.
    """
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        shape = action_space.shape
        assert len(shape) == 1
        if shape[0] == 1:
            return 'continuous'
        else:
            return 'multi_continuous'
    elif isinstance(action_space, gym.spaces.Discrete):
        return 'discrete'
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return 'multi_discrete'
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return 'multi_binary'
    else:
        raise NotImplementedError
        
def get_action_spec(env_action_type: str, env: gym.Env):
    if env_action_type == 'continuous':
        action_dim = 1
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'discrete':
        action_dim = 1
        max_action = env.action_space.n - 1
        min_action = 0
    elif env_action_type == 'multi_continuous':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'multi_discrete':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.nvec.max()
        min_action = env.action_space.nvec.min()
    elif env_action_type == 'multi_binary':
        action_dim = env.actoin_space.n
        max_action = 1
        min_action = 0
    else:
        raise NotImplementedError
    return action_dim, min_action, max_action

def get_state_spec(env: gym.Env):
    if isinstance(env.observation_space, gym.spaces.Discrete):
        state_dim = 1
        max_state = env.observation_space.n - 1
    else:
        state_dim = env.observation_space.shape[0]
        max_state = np.inf
    return state_dim, max_state