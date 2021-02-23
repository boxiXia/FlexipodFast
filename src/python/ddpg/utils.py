#https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
# utils.py
import numpy as np
import gym
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.2, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    
    def __init__(self, env):
        super().__init__(env)
        # assume action_space does not change
#         self._action_space_low = self.action_space.low
#         self._action_space_high = self.action_space.high
        self.act_k = (self.action_space.high - self.action_space.low)/ 2.
        self.act_b = (self.action_space.high + self.action_space.low)/ 2.
        self.act_k_inv = 2./(self.action_space.high - self.action_space.low)
    
    def action(self, action):
        return self.act_k * action + self.act_b

    def reverse_action(self, action):
        return self.act_k_inv * (action - self.act_b)
        

# class Memory:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.buffer = deque(maxlen=max_size)
    
#     def push(self, state, action, reward, next_state, done):
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []

#         batch = random.sample(self.buffer, batch_size)

#         for experience in batch:
#             state, action, reward, next_state, done = experience
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)
        
#         return state_batch, action_batch, reward_batch, next_state_batch, done_batch

#     def __len__(self):
#         return len(self.buffer)
    
class Memory:
    def __init__(self, max_size,state_size,action_size):
        self.max_size = max_size
        self.state_buf = np.zeros((max_size,state_size))
        self.action_buf = np.zeros((max_size,action_size))
        self.reward_buf = np.zeros((max_size,1))
        self.next_state_buf = np.zeros((max_size,state_size))
        self.done_buf = np.zeros(max_size,dtype=bool)
        self.index = 0
    
    def push(self, state, action, reward, next_state, done):
        k = self.index%self.max_size
        self.state_buf[k] = state
        self.action_buf[k] = action
        self.reward_buf[k] = reward
        self.next_state_buf[k] = next_state
        self.done_buf[k] = done
        self.index+=1

    def sample(self, batch_size):
        ids = np.random.choice(len(self),batch_size)
        state_batch = self.state_buf[ids]
        action_batch = self.action_buf[ids]
        reward_batch = self.reward_buf[ids]
        next_state_batch = self.next_state_buf[ids]
        done_batch = self.done_buf[ids]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return self.index if self.index<self.max_size else self.max_size