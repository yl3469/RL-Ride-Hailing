#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:42:16 2021

@author: yujiazhang
@code modified from
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import os, sys
import gym
import numpy as np
import copy
import torch
import random
import math
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path + "/RideHailing/envs/")
import RideHailing
from RideHailing.envs import *
from RideHailing.envs.RideHailing_env import *

env = gym.make('RideHailing-v0', config=CONFIG)

wandb.login(key='74f441f8a5ff9046ae53fad3d92540a168c6bc83')
wandb.init(project='RL', tags=['DQN_FirstTrial'])


# https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
DQN_config = {
    'replay_size': 16384,
    'min_replay_size': 2048,
    'epsilon': 0.8,
    'epsilon_decay_param':50000, 
    'epsilon_decay_power': 0.6,
    'target_update_freq': 200, #update target Q network every x steps
    'sample_replay_size': 1024,
    'discount': 1, 
    'lr': 0.005,
    'step': 20,
    'gamma': 0.2,
    'nstep': 10
    }

#Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# 5/12 TODO: making all the typing consistent

class DQN(nn.Module):
# modified based on previous value network in RH_PPO
# Input is n state, output is n*25 action-values
    def __init__(self, input_size, embedding_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size = 25):
        super(DQN, self).__init__()
        # TODO: What is "L2 regularization over embedding layers"? This is 
        self.embedding_layer = nn.Embedding(361, embedding_size)
        
        assert (hidden_size_1 == input_size - 1 + embedding_size)
        
        self.hidden_layers_stack = nn.Sequential(
            #nn.Linear(embedded_size, hidden_size_1),
            #nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            # CHANGED
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size)
            )


    def forward(self, x):
        #import pdb; pdb.set_trace()
        if x.dim() == 1:
            epoch_embedded = self.embedding_layer(x[0].to(torch.int64))
            #out = torch.cat([epoch_embedded, torch.tensor(x[1:])])
            out = torch.cat([epoch_embedded, x[1:]])
            out = out.type(self.hidden_layers_stack[0].weight.dtype)
            out = self.hidden_layers_stack(out)
        else:
            epoch_embedded = self.embedding_layer(x[:,0].to(torch.int64))
            #out = torch.cat([epoch_embedded, torch.tensor(x[:,1:])])
            outout = torch.cat((torch.transpose(epoch_embedded,0,1), torch.transpose(x[:,1:],0,1)))
            outout = outout.type(self.hidden_layers_stack[0].weight.dtype)
            out = self.hidden_layers_stack(torch.transpose(outout, 0, 1))

        return out


class DQN_Agent:
    def __init__(self, config=DQN_config):
        
        # initialize main model and target model
        self.model = DQN(456, 6, 461, 44, 5)
        self.target_model = DQN(456, 6, 461, 44, 5)
                
        # initialize replay buffer
        self.replay_memory = deque(maxlen=config['replay_size'])
        
        self.nstep_buffer= []
        self.nstep = config['nstep']
        
        # track when to update target network's weights with main's
        self.target_update_counter = 0
        wandb.watch(self.model, log='all')
        
        #self.current_lr = config['lr']
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = config['lr'])
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['step'], gamma=config['gamma'])

        self.training_loss = []

          
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def prep_nstep_transition(self, current_state, action_1D, reward, new_state, done, config=DQN_config):
        self.nstep_buffer.append((current_state, action_1D, reward, new_state, done))
        
        if(len(self.nstep_buffer)<self.nstep):
            return
        
        R = sum([self.nstep_buffer[i][2]*(config['discount']**i) for i in range(self.nstep)])
        
        old_state, old_action, _, _, _ = self.nstep_buffer.pop(0)
        
        self.replay_memory.append((old_state, old_action, R, new_state, done))
        
        #return old_state, old_action, R, new_state, done


    def train(self, config = DQN_config):

        if len(self.replay_memory) < config['min_replay_size']:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, config['sample_replay_size'])
        
        self.model.eval()
        self.target_model.eval()
        
        # Sample states from minibatch, then query NN model for Q values
        # transition is the tuple (state, action, reward, new_state, done)
        #import pdb; pdb.set_trace()
        current_states =  [minibatch[i][0] for i in range(len(minibatch))]
        current_states = torch.stack(current_states) # Change list to tensor
        current_qs_list = self.model(current_states)
        
        # Get future states from minibatch, then query target Q model for Q values
        new_current_states = [minibatch[i][3] for i in range(len(minibatch))]
        new_current_states = torch.stack(new_current_states)
        future_qs_list = self.target_model(new_current_states)
        
        #X = []
        y = []

        # Now we need to enumerate our batches
        # (current_state,action) is <nstep> ahead of new_current_state
        # reward is the rewards accumulated over <nstep> transitions
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # In the replay buffer, states are tensors and actions are 1D (already converted)    

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                # TODO: question -- do we want to restrict the action's origin when taking the argmax
                #prev_action_origin = action // 5
                #max_future_q = torch.max(future_qs_list[index][5*prev_action_origin:(5*(prev_action_origin+1))])
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + config['discount']**self.nstep * float(max_future_q)
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = copy.deepcopy(current_qs_list[index].detach())
            current_qs[action] = new_q

            # And append to our training data
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # train self.model to fit the data in the batch in the pytorch framework
        self.model.train()

        criterion = nn.SmoothL1Loss() 
        y=torch.stack(y)
        # we want to minimize the TD error
        loss = criterion(current_qs_list, y) 
        
        
        # Optimize the model
        #optimizer = optim.Adam(self.model.parameters(), lr = config['lr'])
        #optimizer = optim.Adam(self.model.parameters(), lr = self.current_lr)
        
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step'], gamma=config['gamma'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        #config['lr'] = optimizer.param_groups[0]['lr']
        #import pdb; pdb.set_trace()
        #self.current_lr = optimizer.param_groups[0]['lr']
        

        wandb.log({"loss_Q_network": loss})
        
        
        self.training_loss.append(loss.item())

        # Update target network counter every episode
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > config['target_update_freq']: #L
            # copy the weights of the current main model to the target model
            self.target_model = copy.deepcopy(self.model)
            self.target_update_counter = 0
            # reset learning rate to initial undecayed value after target update
            #self.optimizer = optim.Adam(self.model.parameters(), lr = config['lr'])
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['step'], gamma=config['gamma'])
            #print('current learning rate {}'.format(self.optimizer.param_groups[0]['lr']))
            #print("update target; current loss for Q network {}".format(loss.item()))
            #print(loss.item())
    

def run_DQN(env, num_training_episodes, num_eval_episodes, config = DQN_config):
    agent = DQN_Agent()
    frac_filled_record = []

    for i in range(num_training_episodes):
        #episode_reward = 0
        step_counter = 1
        
        env.reset()
        current_state_raw = env.state # need to process
        current_state = process_raw_state(current_state_raw)
        done = False
        
        
        while not done:
            # epsilon-greedy
            action = np.zeros(2)
            action[0] = int(env.state['origin'])
            
            #epsilon = config['epsilon'] * np.exp(-step_counter/config['epsilon_decay_param'])
            epsilon = config['epsilon'] * step_counter**(-config['epsilon_decay_param'])
                        
            if np.random.random()>epsilon:
                action[1] = int(torch.argmax(agent.model(current_state)[5*int(action[0]):(5*(int(action[0])+1))]))
            else:
                action[1] = np.random.randint(5) 
                        
            action = action.astype(int)
            
            new_state_raw, reward, done, _ = env.step(action)
            new_state = process_raw_state(new_state_raw)
            #episode_reward += reward
            
            action_1D = action[0] * 5 + action[1]
            
            agent.prep_nstep_transition(current_state, action_1D, reward, new_state, done)
            
            #old_s, old_a, nstep_r, new_s, done = agent.prep_nstep_transition(current_state, action_1D, reward, new_state, done)
            #agent.update_replay_memory((old_s, old_a, nstep_r, new_s, done))
            #agent.update_replay_memory((current_state, action_1D, reward, new_state, done))
            
            agent.train()
            # stop and evaluate
            # TODO: we should separate into training/validation for early stopping
            if (step_counter > 10*config['target_update_freq'] and np.mean(np.array(agent.training_loss[-10:]))  < 0.02):
                break
            current_state = copy.deepcopy(new_state)
            step_counter += 1
            
            if step_counter % 1000 == 0:
                print('Running episode {}, simulation step {}'.format(i, step_counter))
        
        #frac_filled = episode_reward / (episode_reward + env.state['unfilled_requests'])
        #frac_filled_record.append(frac_filled)
        # CHECK
        agent.optimizer = optim.Adam(agent.model.parameters(), lr = config['lr'])
        agent.scheduler = optim.lr_scheduler.StepLR(agent.optimizer, step_size=config['step'], gamma=config['gamma'])

    
    agent.model.eval()
    
    for j in range(num_eval_episodes):
        episode_reward = 0
        env.reset()
        
        current_state_raw = env.state # need to process
        #current_state = process_raw_state(current_state_raw)
        done = False
        sim_step_counter=0
        
        while not done:
            # run epsilon-greedy on learned model
            action = np.zeros(2)
            action[0] = int(env.state['origin'])
            
            epsilon = config['epsilon'] * np.exp(-sim_step_counter/config['epsilon_decay_param'])
                        
            if np.random.random()>epsilon:
                action[1] = int(torch.argmax(agent.model(current_state)[5*int(action[0]):(5*(int(action[0])+1))]))
            else:
                action[1] = np.random.randint(5) 
                        
            action = action.astype(int)
            
            new_state_raw, reward, done, _ = env.step(action)
            #new_state = process_raw_state(new_state_raw)
            episode_reward += reward
            
            sim_step_counter += 1
            
            if sim_step_counter % 1000 == 0:
                print('Running sim episode {}, simulation step {}'.format(j, sim_step_counter))
        
        frac_filled = episode_reward / (episode_reward + env.state['unfilled_requests'])
        frac_filled_record.append(frac_filled)
        print('sim episode {} finished, frac filled {}'.format(j, frac_filled))
        wandb.log({"DQN_eval_episode_reward": episode_reward})
        wandb.log({"DQN_eval_episode_frac_filled": frac_filled})
    
    frac_filled_array = np.asarray(frac_filled_record)
    
    print(frac_filled_array)
    
    return np.mean(frac_filled_array), np.std(frac_filled_array)


def process_raw_state(raw_state):
    state_info = np.concatenate((np.asarray([raw_state['epoch']]), 
                                 raw_state['cars'].reshape(-1), 
                                 raw_state['do_nothing'].reshape(-1), 
                                 raw_state['passengers'].reshape(-1)))

    return torch.tensor(state_info)

