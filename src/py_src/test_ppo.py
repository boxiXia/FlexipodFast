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

from torch.utils.tensorboard import SummaryWriter

# from ddpg.utils import NormalizedEnv
from flexipod_env import FlexipodEnv


############## Hyperparameters ##############
# env_name = "BipedalWalker-v3"
env_name = "flexipod"
render = True
# solved_reward = 1500        # stop training if avg_reward > solved_reward
# log_interval = 20           # print avg reward in the interval
# log_interval = 2           # print avg reward in the interval


update_timestep = 4000      # update policy every n timesteps

# max_episodes = 4000                   # max training episodes
max_timesteps = 125*update_timestep   # max training timesteps
max_episode_timesteps = 1500 # max timesteps in one episode



# action_std = 1.0            # constant std for action distribution (Multivariate Normal)
action_std = 0.5            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 0.0002                 # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = 42
#############################################
# creating environment
num_sensors = 128
num_obs = 5
step = 4 # UDP stepping
env = FlexipodEnv(num_sensors = num_sensors,num_observation=num_obs)
# env = gym.make(env_name)
# state_dim = env.observation_space.shape[0]
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.shape[0]

########################################################
for trial in range(3):
    folder_name = f"runs/soft@{1}_internal@{4}_num@{num_sensors}_loc@all_obs@{num_obs}_step@{step}_trial@{trial}"
    writer = SummaryWriter(folder_name)

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
    time_step = 0 # total time step
    i_episode = 0
    t = 0
    episode_reward_list = []
    episode_length_list = []

    # checkpoint = ppo.load(f'{folder_name}/PPO_continuous_{env_name}_best.pth')
    # checkpoint = ppo.load(f'./PPO_continuous_{env_name}.pth')
    # max_avg_length = checkpoint["avg_length"]

    # training loop
    state = env.reset()
    for time_step in range(1,max_timesteps+1):
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # if render:
            #     env.render()
            running_reward += reward
            t+=1
            if done:
                # print(f"done #{i_episode}")
                # log episode info
                episode_length_list.append(t)
                episode_reward_list.append(running_reward)
                print(f'Episode {i_episode} \t length: {t:.0f} \t reward: {running_reward:.0f}')
                i_episode+=1
                # reset
                t = 0 
                running_reward = 0
                state = env.reset()

            # update if its time
            if time_step % update_timestep == 0:
                env.pause()# pause the simulation
                print(f"update #{time_step}")
                ppo.update(memory)
                memory.clear_memory()

                # logging
                avg_episode_length = np.average(episode_length_list[-5:])
                avg_episode_reward = np.average(episode_reward_list[-5:])

                writer.add_scalar("avg_episode_length/train", avg_episode_length, time_step)
                writer.add_scalar("avg_episode_reward/train", avg_episode_reward, time_step)
                if avg_episode_length>max_avg_length:
                    max_avg_length = avg_episode_length
                    ppo.save(f'{folder_name}/PPO_continuous_{env_name}_best.pth',avg_length=max_avg_length)
                elif np.random.random()>2*avg_episode_length/max_avg_length:# % chance 
                    checkpoint = ppo.load(f'{folder_name}/PPO_continuous_{env_name}_best.pth')
                    print(f"load old best,avg_length={checkpoint['avg_length']}")# restart

                env.resume()# resume the simulation
            
env.pause()