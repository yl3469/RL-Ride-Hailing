#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 15:21:58 2021

@author: yujiazhang, yueyingli
"""
from wandb import config
import wandb
import os, sys
import time
import gym
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
wandb.login(key='74f441f8a5ff9046ae53fad3d92540a168c6bc83')
wandb.init(project='RL', tags=['FirstTrail'])

module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path + "/RideHailing/envs/")
import RideHailing
from RideHailing.envs import *
from RideHailing.envs.RideHailing_env import *

env = gym.make('RideHailing-v0', config=CONFIG)

PPO_config = {
    #"initial_theta": 
    "J": 75, # num policy iterations
    "K": 1, # num episodes in Monte Carlo
    "clipping": 0.2,
    "LR_policy": 0.00005, # learning rate for policy network
    "LR_value": 0.0001, # learning rate for value network
    "num_epochs_policy": 3,
    "num_epochs_value": 10,
    "target_KL": 0.012,
    "L2_coeff": 0.005,
    "batch_size": 4096,
    "batch_size_policy": 8, # CHECK
    "num_eval_iters": 4
    }

"""
TODO:
    1. check correctness of train_policy_network()
    2. add KL early stopping to policy network training
    3. add L2 regularization to epoch embedding
    4. increase num Monte Carlo episodes
"""
    
    
sim_output = []
    
    
def run_PPO(env, config = PPO_config):
    
    sim_output = []

    old_policy_model = PolicyNet(456, 6, 461, 44, 5)
    
    policy_training_performance = []

    for j in range(config['J']):
        start_policy_iteration_time = time.time()

        # STEP 1: run K episodes of Monte Carlo using current policy
        sim_output = run_ridehailing_sims(env, old_policy_model)
        print('simulation step finished; starting sim output data processing')
        #print(sim_output)
        # STEP 2: compute value estimate
        V_hat =  compute_V_hat(sim_output)
        #print(V_hat)
        # process V_hat to get a value vector (np array)
        value_est_vector = np.array([item for sublist in V_hat for item in sublist])
        # turn this into tensor
        value_est_tensor = torch.tensor(value_est_vector).float()

        # process sim_output to get state as np array and tensor
        state_array = np.array([sim_output[k][i][0] for k in range(len(sim_output)) for i in range(len(sim_output[k])) ])
        state_tensor = torch.tensor(state_array).float()
        
        print('Value est and state array processed; starting value training')
        
        # (should be length of sim_output[k][i][0])
        assert(value_est_vector.shape[0] == state_array.shape[0]), 'value estimates array and state vector array should be the same size'

        # STEP 3: learn func approximator for value
        # TODO: do not hardcode layer sizes
        value_model = ValueNet(456, 6, 461, 44, 5)
        value_model.train()
        
        # normalize data and get mean, std before passing into training func
        #value_normalizer = Scaler(state_tensor, value_est_tensor)
        #state_std, state_mean = value_normalizer.getx()
        #value_std, value_mean = value_normalizer.gety()
        state_part_mean = torch.mean(state_tensor[:,1:], axis=0)
        state_part_std = torch.std(state_tensor[:,1:], axis=0)
        # CHECK
        state_norm_input = torch.cat((state_tensor[:,0].unsqueeze(1), (state_tensor[:,1:]-state_part_mean)/state_part_std), axis=1)
        idx = np.isnan(state_norm_input)
        # import pdb; pdb.set_trace()
        state_norm_input[idx] = 0
        #state_tensor_norm = (state_tensor - state_mean) / state_std
        # torch.Size([56949, 456])

        #import pdb; pdb.set_trace()
        #value_est_tensor_norm = (value_est_tensor - value_mean) / value_std
        
        new_value_model = train_value_network(state_norm_input, value_est_tensor, value_model)
        #new_value_model = train_value_network(state_tensor, value_est_tensor, value_model)

        print('Value model trained; starting policy training')
        
        value_model = copy.deepcopy(new_value_model)
        
        # STEP 4: compute advantage estimates
        value_model.eval()
        A_hat = compute_A_hat(sim_output, value_model, state_part_mean, state_part_std)
        
        # STEP 5: compute PPO surrogate objective and update policy
        policy_model = copy.deepcopy(old_policy_model)

        policy_model.train()
        new_policy_model = train_policy_network(policy_model, old_policy_model, A_hat, j, state_part_mean, state_part_std)
        policy_model = copy.deepcopy(new_policy_model)
    
        print('Policy model trained; starting evaluation')
    
        # STEP 6: run the updated policy for some number of iterations
        # to get performance, which we need for plotting the learning curve
        _, evaluation_output = run_ridehailing_sims(env, policy_model, num_episodes = config['num_eval_iters'], evaluate = True)
        policy_training_performance.append(evaluation_output)
        
        print('Policy iteration {} evaluation result'.format(j), evaluation_output)
        print('========================================')
        wandb.log({'policy_iteration': j, 'time_iteration': time.time() - start_policy_iteration_time, 'rec_met_mean': evaluation_output[0], 'rec_met_std': evaluation_output[1]})
        old_policy_model = copy.deepcopy(policy_model)
      
    # return data for learning curve
    # can we store this?
    return policy_training_performance
    

def run_ridehailing_sims(env, policy_model, num_episodes = PPO_config['K'], evaluate = False, verbose = False):
    """
    Runs ridehailing simulation

    Parameters
    ----------
    env : Environment
    policy_model : policy mapping state to action
    num_episodes : Number of Monte Carlo episodes to run. The default is PPO_config['K'].
    evaluate : boolean. If true, also output evaluation results (fraction of requests filled)

    Returns
    -------
    simulation output (list of (state, action, 1-step reward))
    optionally, returns policy evaluation results

    """
    
    sim_output = [[None for x in range(1)] for y in range(num_episodes)]  
    frac_requests_met = []
    
    for k in range(num_episodes):
        print('starting episode {}'.format(k))
        env.reset()
        episode_over = False
        cum_reward = 0
        while (episode_over != True):
            if verbose: 
                print('Episode {}, epoch {}, SDM clock {}, origin{}'.format(k, env.state['epoch'], env.state['SDM_clock'], env.state['origin']))
        
            state_input = np.concatenate((np.asarray([env.state['epoch']]), env.state['cars'].reshape(-1), env.state['do_nothing'].reshape(-1), env.state['passengers'].reshape(-1)))

            action_prob_dist = policy_model(torch.tensor(state_input).float())
            action_prob_dist_2D = action_prob_dist.reshape((5,5))
            # Question: why we don't make action_prob_dist as 5 dim? 
            dest_taken = random.choices(range(5), action_prob_dist_2D[env.state['origin']])[0] #25 dim -> scalar
            # map the action_taken (0,..,24) to [origin,dest] to feed into simulator
            #action_taken_tuple = np.array([action_taken//5, action_taken%5])
            action_tuple = np.array([env.state['origin'], dest_taken])
            new_state, reward, episode_over, _ = env.step(action_tuple)
            cum_reward += reward
            
            action_taken_idx = 5 * env.state['origin'] + dest_taken
            state_info = np.concatenate((np.asarray([new_state['epoch']]), new_state['cars'].reshape(-1), new_state['do_nothing'].reshape(-1), new_state['passengers'].reshape(-1)))
            sim_output[k].append([state_info, action_taken_idx, reward])
            #state_array = np.vstack((state_array, state_info))
        
        del sim_output[k][0]
        
        
        unfilled_requests = env.state['unfilled_requests']
        frac_requests_met.append(cum_reward/(cum_reward+unfilled_requests))        
        
    frac_requests_met_array = np.asarray(frac_requests_met)
    
    if evaluate:
        return sim_output, [np.mean(frac_requests_met_array), np.std(frac_requests_met_array)]
    else:
        return sim_output

def compute_V_hat(sim_output):
    """
    Computes value estimates 

    Parameters
    ----------
    sim_output : list of (state, action, 1-step reward) from Monte Carlo sims

    Returns
    -------
    V_hat : list of value estimates. 
            V_hat[k][i] is a scalar, the value estimate of state i in episode k
    """
    
    V_hat = [[None for x in range(1)] for y in range(len(sim_output))]
    
    for k in range(len(sim_output)):
        
        value_cum_backward = 0 
        value_backward_list = []
        
        # compute the value estimates (cum.reward) backwards
        # incrementing by an additional step of reward at a time
        for i in reversed(range(len(sim_output[k]))):
            value_cum_backward += sim_output[k][i][2]
            value_backward_list.append(value_cum_backward)
        
        # reverse the order and store into V_hat
        value_backward_list.reverse()
        V_hat[k] = copy.deepcopy(value_backward_list)
    
    return V_hat
    #V_hat[k][i] is a scalar


def compute_A_hat(sim_output, value_model, 
                  state_part_mean,
                  state_part_std):
    """
    Computes advantage estimates for simulation states

    Parameters
    ----------
    sim_output : list of Monte Carlo simulation output
    value_model : value function approximator

    Returns
    -------
    A_hat : list of (state, action, advantage_estimate) for each step in the sim

    """

    A_hat = [[None for x in range(1)] for y in range(len(sim_output))]
    
    for k in range(len(sim_output)):
        
        for i in range(len(sim_output[k])):
            
            curr_action_idx = sim_output[k][i][1]
            curr_state = sim_output[k][i][0]
            curr_state_tensor = torch.tensor(curr_state).float()
            curr_state_norm_part = (curr_state_tensor[1:]-state_part_mean)/state_part_std
            curr_state_norm_part[np.isnan(curr_state_norm_part)] = 0

            curr_state_norm_input = torch.cat((curr_state_tensor[0].unsqueeze(0), curr_state_norm_part))
            
            if i < len(sim_output[k]) - 1:
                # if curr_state is not yet the last state of the episode
                next_state = sim_output[k][i+1][0]
                next_state_tensor = torch.tensor(next_state).float()
                next_state_norm_input = torch.cat((next_state_tensor[0].unsqueeze(0), (next_state_tensor[1:]-state_part_mean)/state_part_std))
                next_state_norm_input[np.isnan(next_state_norm_input)] = 0

                #curr_state_value_est_norm = value_model(curr_state_tensor_norm)
                #next_state_value_est_norm = value_model(next_state_tensor_norm)
                
                #curr_state_value_est = curr_state_value_est_norm * value_std + value_mean
                #next_state_value_est = next_state_value_est_norm * value_std + value_mean
                
                #normalizer = Scaler(curr_state_tensor, next_state_tensor)
                # Here x, y denotes the cur_state and next_state
                #x_std, x_mean = normalizer.getx()
                #y_std, y_mean = normalizer.gety()
                #x_train_norm = (curr_state_tensor-x_mean)/ x_std
                #y_train_norm = (next_state_tensor-y_mean)/ y_std

                curr_state_value_est = value_model(curr_state_norm_input)
                next_state_value_est = value_model(next_state_norm_input)
        
                adv_est = sim_output[k][i][2] + next_state_value_est - curr_state_value_est
              
            else:
                # if curr_state is the last state of the episode
                adv_est = sim_output[k][i][2]
            
            A_hat[k].append([curr_state, curr_action_idx, adv_est])
        
        del A_hat[k][0]

    return A_hat


class ValueNet(nn.Module):
    """
    Class for value function approximator
    """
    def __init__(self, input_size, embedding_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size = 1):
        super(ValueNet, self).__init__()

        # TODO: What is "L2 regularization over embedding layers"? This is 
        self.embedding_layer = nn.Embedding(361, embedding_size)
        
        #embedded_size = input_size - 1 + embedding_size
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

    def forward(self, x):
        
        #epoch_embedded = self.embedding_layer(torch.tensor(int(x[0])))
        # x [4096, 456]
        #import pdb; pdb.set_trace()
        #epoch_embedded = self.embedding_layer(x[:,0].to(torch.int64))
        #out = torch.cat((epoch_embedded, torch.tensor(x[:,1:])), dim = 0)
        #outout = torch.cat((torch.transpose(epoch_embedded,0,1), torch.transpose(x[:,1:],0,1)))
        #out = self.hidden_layers_stack(torch.transpose(outout, 0, 1))

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



class PolicyNet(nn.Module):
    """
    Class for policy function approximator
    """
    def __init__(self, input_size, embedding_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size = 25):
        super(PolicyNet, self).__init__()

        # TODO: What is "L2 regularization over embedding layers"?
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
        self.softmax= nn.Softmax(dim=None)
        
        
    def forward(self, x):
        # TODO: check the way we deal with embedding is correct
        #import pdb; pdb.set_trace()
        if x.dim() == 1:
            epoch_embedded = self.embedding_layer(x[0].to(torch.int64))
            out = torch.cat([epoch_embedded, torch.tensor(x[1:])])
            out = self.hidden_layers_stack(out)
            out = self.softmax(out)
        else:
            epoch_embedded = self.embedding_layer(x[:,0].to(torch.int64))
            #out = torch.cat([epoch_embedded, torch.tensor(x[:,1:])])
            outout = torch.cat((torch.transpose(epoch_embedded,0,1), torch.transpose(x[:,1:],0,1)))
            out = self.hidden_layers_stack(torch.transpose(outout, 0, 1))
            out = self.softmax(out)

        
        return out
    
    

def train_value_network(x_train, y_train, value_model, config = PPO_config):
    """
    Trains value function approximator given Monte Carlo states and value estimates
    Called in each policy iteration
    
    Parameters
    ----------
    x_train : numpy array of states in Monte Carlo simulation
    y_train : numpy array of value function estimates of each state in sims
    value_model : initial value function approximator
    config : dictionary. The default is PPO_config.

    Returns
    -------
    value_model : trained value function approximator
    """
    value_loss = torch.nn.MSELoss()
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr = config['LR_value'])

    #normalizer = Scaler(x_train, y_train)
    #x_std, x_mean = normalizer.getx()
    #y_std, y_mean = normalizer.gety()
    #x_train_norm = (x_train-x_mean)/ x_std
    #y_train_norm = (y_train-y_mean)/ y_std
    
    batch_size = config['batch_size']


    for epoch in range(config['num_epochs_value']):
        #shuffle the training dataset (it's the same training dataset, just shuffled)
        random_idx = torch.randperm(x_train.size()[0])
        #x_train_norm_shuff = x_train_norm[random_idx,]
        #y_train_norm_shuff = y_train_norm[random_idx]
        x_train_norm_shuff = x_train[random_idx,]
        y_train_norm_shuff = y_train[random_idx]
        
        
        # forward and backward passes in batches
        for batch in range(math.ceil(x_train.size()[0] / batch_size)):
            # get the current batch of training data (x and y)
            x_train_norm_batch = x_train_norm_shuff[(batch*batch_size):((batch+1)*batch_size),]
            #import pdb; pdb.set_trace()
            y_train_norm_batch = y_train_norm_shuff[(batch*batch_size):((batch+1)*batch_size)]
            
            value_optimizer.zero_grad()
            y_train_pred_norm = value_model(x_train_norm_batch)
            loss = value_loss(y_train_pred_norm.squeeze(), y_train_norm_batch)
            loss.backward()
            value_optimizer.step()
        wandb.log({"epoch_value_net": epoch, "loss_value_net": loss})
    
    return value_model



def train_policy_network(policy_model, old_policy_model, A_hat, j, state_part_mean, state_part_std, config = PPO_config):
    """
    Trains policy model mapping state to action (trip)

    Parameters
    ----------
    policy_model : policy to be trained
    old_policy_model : policy from previous iteration
    A_hat : list of (state, action, advantage_estimate) for each step in the sim
    j : index of policy iteration
    config : default is PPO_config.

    Returns
    -------
    policy_model : Trained policy model

    """
    
    # TODO: How to implement KL early stopping? 
    
    learning_rate = np.amax([0.01, 1-j/config['J']]) * config['LR_policy']
    #PPO_loss = compute_surr_objective(A_hat, old_policy_model, policy_model, j, PPO_config)        
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr = learning_rate)
    
    
    for epoch in range(config['num_epochs_policy']):
        
        # we shuffle the A_hat entries and divide them into batches
        # full size of A_hat = K = num Monte Carlo sims
        random.shuffle(A_hat)
        
        batch_size = config['batch_size_policy']
        
        for batch in range(math.ceil(len(A_hat) / batch_size)):
            print('training policy epoch {}, batch {}'.format(epoch, batch))

            policy_optimizer.zero_grad()
            loss = compute_surr_objective(A_hat[(batch*batch_size):((batch+1)*batch_size)], old_policy_model, policy_model, j, state_part_mean, state_part_std, PPO_config)
            loss.backward(retain_graph = True)
            policy_optimizer.step()
            wandb.log({"batch_policy_network": batch, "loss_policy_network": loss})
         
    return policy_model







def compute_surr_objective(A_hat, old_policy_model, policy_model, j, state_part_mean, state_part_std, config=PPO_config):
    """
    Computes clipped surrogate objective for PPO optimization
    (Clipping param decays with num of policy iteration)

    Parameters
    ----------
    A_hat : list of (state, action, advantage_estimate) for each step in the sim
    old_policy_model : policy from previous iteration
    policy_model : policy to be trained
    j : index of policy iteration
    config : default is PPO_config.

    Returns
    -------
    surr_objective: clipped PPO surrogate objective

    """
    
    def clip(c, epsilon):
        if c < 1-epsilon:
            return 1- epsilon
        elif c > 1+epsilon:
            return 1+ epsilon
        else:
            return c
    
    clipping_param = np.amax([0.01, (1-j/config['J'])*config['clipping']])
    #K = PPO_config['K']
    #assert(K==len(A_hat))

    surr_objective = 0
    for k in range(len(A_hat)):
        for i in range(len(A_hat[k])):
            # CHECK: if the action is an integer corresponding to the state
            curr_state = torch.tensor(A_hat[k][i][0]).float()
            try:
                curr_state_norm_input = torch.cat((curr_state[:,0].unsqueeze(1), (curr_state[:,1:]-state_part_mean)/state_part_std))
            except:
                curr_state_norm_input = torch.cat((curr_state[0].unsqueeze(0), (curr_state[1:]-state_part_mean)/state_part_std))

            curr_state_norm_input[np.isnan(curr_state_norm_input)] = 0
            curr_action = A_hat[k][i][1]
            r_theta_ksi = policy_model(curr_state_norm_input)[curr_action] / old_policy_model(curr_state_norm_input)[curr_action]  
            A_hat_ksi = A_hat[k][i][2] # Estimate the A_hat
            surr_objective += min(r_theta_ksi * A_hat_ksi, clip(r_theta_ksi, clipping_param) * A_hat_ksi)
    surr_objective /= len(A_hat)

    return surr_objective




class Scaler:
    def __init__(self, x, y):
        self.x_mean = torch.mean(x, axis=0)
        self.y_mean= torch.mean(y, axis=0)
        self.x_std = torch.std(x, axis=0)
        self.y_std = torch.std(y, axis=0)
        
    def getx(self):
        return self.x_std, self.x_mean # return saved mean and std dev of x

    def gety(self):
        return self.y_std, self.y_mean # return saved mean and std dev of y



def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


