#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:28:28 2021

@author: yujiazhang, yueyingli
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
from copy import deepcopy
## Global Variable
R = 5
N = 1000
H = 360
L = 5 
tau_max = 75

## Init Config
CONFIG = {
    'P': np.array([
        [[0.6, 0.1, 0, 0.3, 0], 
         [0.1, 0.6, 0, 0.3, 0], 
         [0, 0, 0.7, 0.3, 0],
         [0.2, 0.2, 0.2, 0.2, 0.2],
         [0.3, 0.3, 0.3, 0.1, 0]],
        [[0.1, 0, 0, 0.9, 0],
         [0, 0.1, 0, 0.9, 0],
         [0, 0, 0.1, 0.9, 0],
         [0.05, 0.05, 0.05, 0.8, 0.05],
         [0, 0, 0, 0.9, 0.1]],
        [[0.9, 0.05, 0, 0.05, 0],
         [0.05, 0.9, 0, 0.05, 0],
         [0, 0, 0.9, 0.1, 0],
         [0.3, 0.3, 0.3, 0.05, 0.05],
         [0, 0, 0, 0.1, 0.9]]]),
    'tau': np.array([
        [[9, 15, 75, 12, 24],
         [15, 6, 66, 6, 18],
         [75, 66, 6, 60, 39],
         [15, 9, 60, 9, 15],
         [30, 24, 45, 15, 12]],
        [[9, 15, 75, 12, 24],
         [15, 6, 66, 6, 18],
         [75, 66, 6, 60, 39],
         [12, 6, 60, 9, 15],
         [24, 18, 39, 15, 12]],
        [[9, 15, 75, 12, 24],
         [15, 6, 66, 6, 18],
         [75, 66, 6, 60, 39],
         [12, 6, 60, 9, 15],
         [24, 18, 39, 15, 12]]]),
    'lambda_arr': np.array([
        [1.8, 1.8, 1.8, 1.8, 18],
        [12, 8, 8, 8, 2],
        [2, 2, 2, 22, 2]]),
    'rewards': np.array([0, 0, 1]),
    'allow_empty_rerouting': True
}
    

 ## Helper Func
def calc_initial_passenger_demand(arr_rate):

    res = np.zeros(R)
    denom = sum((arr_rate))
    for i in range(R-1):
        res[i] = np.floor(arr_rate[i] / denom * N)
    res[-1] = N - sum(res)
#     assert(sum(res) == N)

    return res

init_passenger_demand = calc_initial_passenger_demand([1.8, 1.8, 1.8, 1.8, 18])


CONFIG['starting_state'] = {
    'epoch': 0,
    'cars': 
        np.concatenate((np.reshape(init_passenger_demand, (-1, 1)), np.zeros((R, tau_max+L-1))) , axis=1),
    'passengers': np.zeros((R,R)),   
    # auxiliary states that help with simulation
    'SDM_clock': 0, #CHECKED: should just be 0 because no arrivals at time 0
    'origin': 0,
    #'do_nothing': np.zeros((R, tau_max+L)),
    'do_nothing': np.zeros((R, L+1)),
    'unfilled_requests': 0
}
    
    
class RideHailingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    starting_state  = dict()
    def __init__(self, config):
        
        # Initializes model parameters based on a configuration dictionary
        
        # P: (3,5,5) array of region-transition-probabilities (corresponding to three time periods)
        # tau: (3,5,5) array of mean travel times (corresponding to three time periods)
        # lambda_arr: 5-dim vector of arrival rates to each region
        # rewards: 3-dim vector of per-trip reward, default should be np.array([0,0,1])
        # R: number of regions
        # N: number of cars
        # H: number of time steps
        # L: patience time
        
        self.P = config['P']
        self.tau = config['tau']
        self.lambda_arr = config['lambda_arr']
        self.rewards = config['rewards']
        
        # tau_max is the max value of all tau tau entries; 
        # this helps us store the cars state in a 2D array
        self.tau_max = np.amax(self.tau)

        self.R = 5
        self.N = 1000 
        self.H = 360 
        self.L = 5
        
        self.allow_empty_rerouting = config['allow_empty_rerouting']

        RideHailingEnv.starting_state = config['starting_state'] 


        # Defines state and action spaces 
        self.observation_space = spaces.Dict({
            'epoch': spaces.Discrete(self.H),
            'cars': 
                spaces.Box(low=0, high=self.N, shape=(self.R, self.tau_max+self.L, 1), dtype=np.uint8), 
            'passengers': spaces.Box(low=0, high=100000, shape = (self.R, self.R), dtype = np.uint8),  #CHECK high 1000 
            'do_nothing': spaces.Box(low=0, high=self.N, shape=(self.R, self.L+1, 1), dtype=np.uint8),
            # auxiliary states that help with simulation
            'SDM_clock': spaces.Discrete(self.N),
            'origin': spaces.Discrete(self.R),
            'unfilled_requests': spaces.Discrete(100000)
            })
                
        # mathematically the action space has hierarchical structure defined by the SDM
        # but that does not matter for the simulation. Can flatten everything
        # i.e., instead of a process of length = horizon, 
        # we have a process of length sum_horizon SDM_length_at_each_horizon
        # each action is an (origin, destination) pair
        
        self.action_space = spaces.Tuple((spaces.Discrete(self.R), spaces.Discrete(self.R)))  
        
        self.state = deepcopy(RideHailingEnv.starting_state)

    # Resets environment to initial state
    def reset(self):
        self.state = deepcopy(RideHailingEnv.starting_state)
        return self.state

    # Defines one step of the MDP, returning the new state, reward, whether time horizon is finished, and a dictionary of information
    def step(self, action):
    
    
        # a constraint on the action should be action[0] = current origin  
        # this should be fulfilled on the agent's part, according to the policy taken
        
        # first map epoch to one of the three time periods 
        epoch = self.state['epoch']
        period = epoch//120
        
        #print('epoch' + str(epoch))
        #print('SDM another step; SDM clock countdown {}'.format(self.state['SDM_clock']))
        
        
        # let's say, 0 for do nothing, 1 for empty rerouting, 2 for a fulfilled request
        trip_type = 0
        
        curr_origin = copy.copy(self.state['origin'])
        #print('SDM current origin: {}'.format(curr_origin))
        
        #print(self.state['cars'])
        
        # We are changing the structure, ensuring there exist available car(s) at state['origin']; 
        # conditions determining whether we should go to the next origin / epoch
        # are enforced after the transitions; i.e., we update state['origin'] by looking one step ahead
        
        # state transition and reward
        trip_type, new_state = self.transition(self.state, action)
        #print('new_state shape: {}'.format(np.shape(new_state['cars'])))
        #print(new_state)
        reward = self.r(trip_type)
        # Note: need shallow copy https://www.programiz.com/python-programming/shallow-deep-copy
        self.state['cars'] = copy.copy(new_state['cars'])
        self.state['passengers'] = copy.copy(new_state['passengers'])
        self.state['do_nothing'] = copy.copy(new_state['do_nothing'])
        self.state['SDM_clock'] -= 1
        
        # decide whether to stay in the current epoch and if so whether to stay in the current origin
        if self.state['SDM_clock']>0 and np.sum(self.state['cars'][curr_origin, :(self.L+1)])>0:
            # SDM countdown is not over; there are still available cars at the current origin
            pass
        elif self.state['SDM_clock']>0 and np.sum(self.state['cars'][curr_origin, :(self.L+1)])==0:
            # SDM countdown is not over; there are no available cars at the current origin
            # loop to next origin with available cars
            new_origin = copy.copy(curr_origin)
            while np.sum(self.state['cars'][new_origin, :(self.L+1)]) == 0:
                new_origin += 1
            self.state['origin'] = min(new_origin, self.R-1) # cap at last region
        
        else:
            # time to move to new epoch
            self.state['epoch'] += 1
            period = np.floor(self.state['epoch']/120).astype(int)
            
            # put the do_nothings back to available cars pool
            for i in range(self.R):
                put_back = self.state['cars'][i]# + self.state['do_nothing'][i]
                put_back[:(self.L+1)] += self.state['do_nothing'][i]
                self.state['cars'][i] = copy.copy(put_back)
                self.state['do_nothing'][i] = np.zeros(self.L+1)
            
            # set new SDM_clock countdown, reset origin
            horizon = 0
            for i in range(self.R):
                horizon += np.sum(self.state['cars'][i, :(self.L+1)])
            self.state['SDM_clock'] = copy.copy(horizon)
            new_origin = 0
            while np.sum(self.state['cars'][new_origin, :(self.L+1)]) == 0:
                new_origin += 1
            self.state['origin'] = min(new_origin, self.R-1)
            
            # decrease etas by 1
            for i in range(self.R):
                
                old_num_cars = copy.copy(self.state['cars'][i])
                #print('old number of cars :{}'.format(old_num_cars))
               
                new_cars = np.pad(old_num_cars[1:], (0, 1), 'constant')
                new_cars[0] += old_num_cars[0]
                self.state['cars'][i] = copy.copy(new_cars)
            
                
            # reset passenger matrix, count number of unfilled requests
            self.state['unfilled_requests'] += np.sum(self.state['passengers'])
            self.state['passengers'] = np.zeros((self.R, self.R))
                
            # simulate new passenger arrivals    
            if period < 3:
                for i in range(self.R):
                    num_arrivals = np.random.poisson(self.lambda_arr[period, i])
                    trips = np.random.multinomial(num_arrivals, self.P[period,i,:]).astype(int)
                    for j in range(self.R):
                        self.state['passengers'][i,j] += trips[j]
        
        
        if epoch >= self.H:
            episode_over = True
        else:
            episode_over = False
        
        #count_num_avail = np.sum(self.state['cars'][:,:(self.L+1)])
        #count_num_unavail = np.sum(self.state['cars'][:,(self.L+1):])
        #print('number of non do-nothing cars: {}'.format(np.sum(self.state['cars'])))
        #print('number of available cars: {}'.format(count_num_avail))
        #print('number of unavailable cars: {}'.format(count_num_unavail))
        #print('number of do-nothing cars: {}'.format(np.sum(self.state['do_nothing'])))
        #print('unfilled requests: {}'.format(self.state['unfilled_requests']))

        return self.state, reward, episode_over, {}



    # Auxiliary function computing the reward
    def r(self, trip_type):
        return self.rewards[trip_type]

    # Auxiliary function computing trip type and transition for a single trip assignment
    def transition(self, state, action):
        curr_origin = copy.copy(state['origin'])
        
        assert action[0] == curr_origin, f'action origin {action[0]} does not align w SDM current origin {curr_origin}'
        
        new_state = copy.copy(state)
        
        # identify first positive entry in the current origin being considered
        car_dist_from_origin = np.min(np.where(state['cars'][curr_origin, :(self.L+1)]>0))
        #print('closest car from origin: {}'.format(car_dist_from_origin))
        # we could be switching to a new period w different traffic parameters
        # when the car starts the next trip
        new_period = np.floor((state['epoch'] + car_dist_from_origin)/120)
        
        car_dist_from_origin = car_dist_from_origin.astype(int)
        new_period = np.min((int(new_period), 2)) # capping the period at 2 in case we reach 360

        # if there are ride requests from current origin to current destination
        if state['passengers'][action[0], action[1]] > 0:
            new_state['passengers'][action[0], action[1]] -= 1
            trip_type = 2
            new_state['cars'][curr_origin, car_dist_from_origin] -= 1
            new_state['cars'][action[1], car_dist_from_origin + self.tau[new_period,curr_origin,action[1]]] += 1
            
        # otherwise (no request), do empty-car-rerouting or do-nothing if empty rerouting allowed
        elif self.allow_empty_rerouting:
            if car_dist_from_origin == 0 and action[1] != curr_origin:
                trip_type = 1
                new_state['cars'][curr_origin, car_dist_from_origin] -= 1
                new_state['cars'][action[1], car_dist_from_origin + self.tau[new_period,curr_origin,action[1]]] += 1
            else:
                trip_type = 0
                new_state['cars'][curr_origin, car_dist_from_origin] -= 1 
                new_state['do_nothing'][curr_origin, car_dist_from_origin] += 1 

        else:
            trip_type = 0
            new_state['cars'][curr_origin, car_dist_from_origin] -= 1 
            new_state['do_nothing'][curr_origin, car_dist_from_origin] += 1 


        return trip_type, new_state
        
    ### NOTES: DYNAMICS of SDM
    
    # if in the middle of SDM, 
        # don't increase epoch number / simulate passenger arrivals / update etas
        # newstate['SDM_clock'] -= 1
        # just update destination for the one car in each SDM update step
        # update trip type
        # everything should be deterministic in this case
    
    # when transitioning into a new epoch
    # here we have randomness due to passenger arrivals    
        # increase epoch number, simulate passenger arrivals, update etas, 
        # compute new countdown value for self.SDM_clock
        # and update destination for the final car in the SDM
        # and update trip type
        # there is stochasticity in this case

