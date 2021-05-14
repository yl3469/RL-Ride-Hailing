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
import torchvision.transforms as T

module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path + "/RideHailing/envs/")
import RideHailing
from RideHailing.envs import *
from RideHailing.envs.RideHailing_env import *

env = gym.make('RideHailing-v0', config=CONFIG)


DQN_config = {
    'replay_size': 4096,
    'min_replay_size': 1024,
    'epsilon': 0.2,
    # think about how to have epsilon decay
    'target_update_freq': 4, #update target Q network every x steps
    'sample_replay_size': 512,
    'discount': 1, 
    'lr': 0.01
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
            nn.Tanh(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.Tanh(),
            nn.Linear(hidden_size_3, output_size)
            )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if x.dim() == 1:
            epoch_embedded = self.embedding_layer(x[0].to(torch.int64))
            out = torch.cat([epoch_embedded, torch.tensor(x[1:])])
            out = self.hidden_layers_stack(out)
        else:
            epoch_embedded = self.embedding_layer(x[:,0].to(torch.int64))
            #out = torch.cat([epoch_embedded, torch.tensor(x[:,1:])])
            outout = torch.cat((torch.transpose(epoch_embedded,0,1), torch.transpose(x[:,1:],0,1)))
            out = self.hidden_layers_stack(torch.transpose(outout, 0, 1))

        return out


class DQN_Agent:
    def __init__(self, config=DQN_config):
        
        # initialize main model and target model
        self.model = DQN()
        self.target_model = DQN()
        
        # initialize replay buffer
        self.replay_memory = deque(maxlen=config['replay_size'])
        
        # track when to update target network's weights with main's
        self.target_update_counter = 0
        
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def train(self, config = DQN_config):
        if len(self.replay_memory) < config['min_replay_size']:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, config['sample_replay_size'])
        
        self.model.eval()
        self.target_model.eval()
        
        # Sample states from minibatch, then query NN model for Q values
        # transition = [state, action, reward, new_state, done]
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model(torch.tensor(current_states))
        
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model(torch.tensor(new_current_states))
        
        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + config['discount'] * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            # TODO: check -- is the copying right?
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # train self.model to fit the data in the batch in the pytorch framework
        self.model.train()
        criterion = nn.SmoothL1Loss() 
        # TODO: check if this is right
        # we want to minimize the TD error
        loss = criterion(current_qs_list, torch.tensor(y)) 
        
        # Optimize the model
        optimizer = optim.Adam(self.model.parameters(), lr = config['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # CHECK: Update target network counter every episode
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > config['target_update_freq']:
            # copy the weights of the current main model to the target model
            self.target_model = copy.deepcopy(self.model)
            self.target_update_counter = 0
    

def run_DQN(num_episodes, config = DQN_config):
    agent = DQN_Agent()
    frac_filled_record = []
    
    for i in range(num_episodes):
        episode_reward = 0
        step = 1
        
        env.reset()
        current_state = env.state # may need to process
        done = False
        while not done:
            # epsilon-greedy
            action = np.zeros(2)
            action[0] = env.state['origin']
            
            if np.random.random()>config['epsilon']:
                action[1] = np.argmax(agent.model(current_state))
            else:
                action[1] = np.random.randint(5) #donot hardcode
            
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # TODO: should we process the state / action as tensor here
            action_1D = action[0] * 5 + action[1]
            #curr_state_tensor
            #new_state_tensor
            
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            #agent.train(done, step)?
            agent.train()
            current_state = copy.deepcopy(new_state)
            step += 1
        
        frac_filled = episode_reward / (episode_reward + env.state['unfilled_requests'])
        frac_filled_record.append(frac_filled)
    
    frac_filled_array = np.asarray(frac_filled_record)
    
    return np.mean(frac_filled_array), np.std(frac_filled_array)



