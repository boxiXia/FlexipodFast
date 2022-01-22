import os
# https://discuss.pytorch.org/t/how-to-change-the-default-device-of-gpu-device-ids-0/1041/24
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{1}"

import pickle as pkl
import time
import sys
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import heapq


# from pytorch_sac.train import Workspace

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_sac")
sys.path.append(p)

from pytorch_sac.utils import setSeedEverywhere, evalMode
from pytorch_sac.replay_buffer import ReplayBuffer
from pytorch_sac.logger import Logger
from pytorch_sac.video import VideoRecorder

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        
        self.model_dir = f"{self.work_dir}//model" # model save dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        setSeedEverywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.env = utils.makeEnv(cfg)
        self.env = hydra.utils.call(cfg.env)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        cfg.agent.n_step = cfg.replay_buffer.n_step # n-step experience replay
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        self.replay_buffer = ReplayBuffer(
            capacity=cfg.replay_buffer.capacity,
            obs_shape = self.env.observation_space.shape,
            action_shape = self.env.action_space.shape,
            obs_dtype = self.env.observation_space.dtype,
            action_dtype = self.env.action_space.dtype,
            n_step = cfg.replay_buffer.n_step, # n-step experience replay
            discount=cfg.agent.discount, # per step discount
            device = self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        print(OmegaConf.to_yaml(cfg))

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            
            with evalMode(self.agent): 
                while not done:
                    action = self.agent.act(obs, sample=False)
                    obs, reward, done, _ = self.env.step(action)
                    self.video_recorder.record(self.env)
                    episode_reward += reward
                    episode_step += 1
                    if episode_step == self.env._max_episode_steps:
                        break # breaks reached max episode stesp
                
            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode', episode, self.step)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step,ty = 'eval')
        return average_episode_reward

    def run(self):
        
        # optimization tricks 
        # https://pytorch.org/docs/stable/backends.html#torch-backends-cudnn
        torch.backends.cudnn.benchmark = True
        
        if self.cfg.load_replay_buffer: # load replay buffer folder path
            self.replay_buffer.load(self.cfg.load_replay_buffer)
        
        num_train_steps = self.cfg.num_train_steps  # total training steps
        num_seed_steps = self.cfg.num_seed_steps  # steps prior to training
        eval_frequency = self.cfg.eval_frequency  # evaluate every x steps
        eval_step = eval_frequency # evaluate if step>eval_step
        update_step = num_seed_steps
        
        replay_buffer_save_interval = int(self.cfg.replay_buffer_save_interval)
        replay_buffer_save_step = replay_buffer_save_interval
        
        env = self.env
        batch_size = self.cfg.agent.batch_size
        update_frequency = self.cfg.update_frequency
                
        # ref: replay buffer emphasizing recent experience
        # https://arxiv.org/abs/1906.04009
        eta_0 = 1
        eta_T = 0.996
        min_sample_size = batch_size*10
        
        # saving top x models
        num_saved_model = self.cfg.num_saved_model
        h_saved_model = [] # container for heapq (priority queue): 
        
        #################################################
        # num_example_steps = num_seed_steps*2        
        
        def computeAction(self,obs):
            action = self.agent.act(obs, sample=True)
            if self.step<num_seed_steps:
                action = env.action_space.sample()
            # with evalMode(self.agent):
            #     action = self.agent.act(obs, sample=True)
            # if self.step< num_example_steps:  # sample action for data collection
            #     if self.step < num_seed_steps:
            #         action = env.action_space.sample()
            #     else: # example step 
            #         action = env.exampleAction()
            return action
                    
        episode = -1        
        # training loop
        while self.step < num_train_steps:
            episode, episode_step, episode_reward, not_done = episode+1, 0, 0.0, True
            start_time = time.time() # for logging episode training duration
            # initialization: reset agent and env
            self.agent.reset()
            obs = env.reset()
            
            # collect episode samples
            with torch.no_grad():
                with evalMode(self.agent):
                    action = computeAction(self,obs)  # a <- pi(s0)
                    env.act(action) # take action
                    
                    while not_done:
                        next_action = computeAction(self,obs)
                        next_obs, reward, done, _ = env.observe()
                        env.act(next_action)
                        
                        max_episode_step_reached = (episode_step + 1 == env._max_episode_steps)
                        not_done_bootstrap = True if max_episode_step_reached else (not done) # allow infinite bootstrap
                        self.replay_buffer.add(obs, action, reward, next_obs, not_done_bootstrap)
                        not_done = not (done or max_episode_step_reached) # signals episode ended
                        
                        ### next->current
                        obs = next_obs
                        action = next_action
                        # increment steps and reward
                        episode_step += 1
                        self.step += 1
                        episode_reward += reward
            
            self.replay_buffer.onEpisodeEnd() # clear n-step buffer on episode end
                       
            self.logger.log('train/duration',time.time() - start_time, self.step)
            self.logger.log('train/episode_reward',episode_reward, self.step)
            self.logger.log('train/episode', episode, self.step)
            
            if (self.step >= num_seed_steps) and (self.step >=batch_size) and (self.step > update_step): # agent update
                num_updates = (self.step-update_step)//update_frequency
                update_step = self.step
                try:env.pause()
                except Exception:pass
                # t1 = time.time()
                
                # compute the annealed eta
                eta = eta_0 + (eta_T-eta_0)* self.step/num_train_steps
                
                for k in range(num_updates): # update several rounds
                    # update using the most recent num_recent samples
                    self.replay_buffer.num_recent = int(max(self.step*eta**(k*1000.0/num_updates),min_sample_size))
                    self.agent.update(self.replay_buffer,self.logger, self.step)
                # print(f"#update:{num_updates},num_recent={self.replay_buffer.num_recent} dt={time.time()-t1:.3f}")

                self.logger.dump(self.step, save=(self.step > num_seed_steps))
            
            # save replay buffer periodically
            if self.step>replay_buffer_save_step:
                replay_buffer_save_step+=replay_buffer_save_interval
                self.replay_buffer.save(f"{self.work_dir}//replay_buffer")
            
            # evaluate agent periodically
            if self.step > eval_step:
                eval_step += eval_frequency
                avg_episode_reward = self.evaluate() # average episode reward
                
                # save the <num_saved_model> top model
                save_path = f"{self.model_dir}//{self.step}_{avg_episode_reward:.0f}.chpt"
                if len(h_saved_model)<num_saved_model:
                    heapq.heappush(h_saved_model,(avg_episode_reward,save_path))
                    self.agent.save(save_path) # saving new
                elif h_saved_model[0][0]<avg_episode_reward: # old reward < current reward
                    avg_episode_reward_pop,save_path_pop = heapq.heapreplace(h_saved_model, (avg_episode_reward,save_path))
                    self.agent.save(save_path) # saving new
                    os.remove(save_path_pop)   # removing old         


@hydra.main(config_path=".", config_name='train')
def main(cfg):
    
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"torch.cuda.current_device()={torch.cuda.current_device()}")
    
    workspace = Workspace(cfg)
    if cfg.load_model:
        workspace.agent.load(cfg.load_model)
    if "train" in cfg and cfg.train:
        workspace.run()
    else:
        workspace.evaluate()
    


if __name__ == '__main__':
    main()
