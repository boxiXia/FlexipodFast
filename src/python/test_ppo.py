import os
# https://discuss.pytorch.org/t/how-to-change-the-default-device-of-gpu-device-ids-0/1041/24
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=f"{1}"

import torch
print(f"current_device:{torch.cuda.current_device()}")
print(f"device_count:{torch.cuda.device_count()}")
device = torch.device("cuda")


import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from ddpg import DDPGagent
from ppo.PPO_continuous import PPO,Memory
# from ddpg.utils import NormalizedEnv
from flexipod_env import FlexipodEnv

env = FlexipodEnv(dof = 12)
# env = NormalizedEnv(env)

############## Hyperparameters ##############
# env_name = "BipedalWalker-v3"
env_name = "flexipod"
render = True
solved_reward = 1500        # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
# log_interval = 2           # print avg reward in the interval

max_episodes = 20000        # max training episodes
max_timesteps = 1500        # max timesteps in one episode

update_timestep = 4000      # update policy every n timesteps
# update_timestep = 300      # update policy every n timesteps


# action_std = 1.0            # constant std for action distribution (Multivariate Normal)
action_std = 0.8            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 0.0002                 # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None
#############################################
# creating environment
# env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

########################################################
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/soft12dof_experiment_2')

##################################
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)

# logging variables
running_reward = 0
avg_length = 0
max_avg_length = 0
time_step = 0

# checkpoint = ppo.load(f'./PPO_continuous_{env_name}_best.pth')
# checkpoint = ppo.load(f'./PPO_continuous_{env_name}.pth')
# max_avg_length = checkpoint["avg_length"]

# training loop
for i_episode in range(0, max_episodes+1):
    state = env.reset()
    for t in range(max_timesteps):
        time_step +=1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if time_step % update_timestep == 0:
            print(f"update #{i_episode}")
            env.pause()# pause the simulation
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
            env.resume()# resume the simulation
        running_reward += reward
        # if render:
        #     env.render()
        if done:
            print(f"done #{i_episode}")
            break

    avg_length += t

    # save every 500 episodes
    if i_episode % 500 == 0:
        ppo.save(f'./PPO_continuous_{env_name}.pth',avg_length=avg_length)

    # logging
    if i_episode % log_interval == 0:
        avg_length = avg_length/log_interval
        running_reward = running_reward/log_interval
        writer.add_scalar("avg_length/train", avg_length, i_episode)
        writer.add_scalar("running_reward/train", running_reward, i_episode)
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            ppo.save(f'./PPO_continuous_solved_{env_name}.pth',avg_length=avg_length)
            break
            
        if avg_length>max_avg_length:
            max_avg_length = avg_length
            ppo.save(f'./PPO_continuous_{env_name}_best.pth',avg_length=avg_length)
        elif np.random.random()<0.1:# 50% chance 
            checkpoint = ppo.load(f'./PPO_continuous_{env_name}_best.pth')
            print(f"load old best,avg_length={checkpoint['avg_length']}")# restart

        print(f'Episode {i_episode} \t Avg length: {avg_length:.0f} \t Avg reward: {running_reward:.0f}')
        running_reward = 0
        avg_length = 0
        
env.pause()