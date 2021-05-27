R = 5
N = 1000
H = 360
L = 5 
import gym
import numpy as np
import sys
import os

module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
sys.path.append(module_path + "/RideHailing/envs/")
print('{}'.format(sys.path))
    
from RideHailing_env import RideHailingEnv


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
    'rewards': np.array([0, 0, 1])
#     'starting_state': { 
#                 'epoch': 0,
#                 'cars': 
#                     spaces.Tuple(
#                     spaces.Box(low=0, high=N, shape=(self.tau_max[0]+L, 1), dtype=np.uint8), #CHECK R not right here -- should it be 1?
#                     spaces.Box(low=0, high=N, shape=(self.tau_max[1]+L, 1), dtype=np.uint8),
#                     spaces.Box(low=0, high=N, shape=(self.tau_max[2]+L, 1), dtype=np.uint8),
#                     spaces.Box(low=0, high=N, shape=(self.tau_max[3]+L, 1), dtype=np.uint8),
#                     spaces.Box(low=0, high=N, shape=(self.tau_max[4]+L, 1), dtype=np.uint8)
#                     ),
#                 'passengers': spaces.Box(low=0, high=1000, shape = (R, R), dtype = np.uint8),  #CHECK high 1000 
#                 # auxiliary states that help with simulation
#                 'SDM_clock': spaces.Discrete(N), #CHECK: remember to initialize it as I_0
#                 'origin': spaces.Discrete(R),
#                 #'destination': spaces.Discrete(R),
#                 'do_nothing': 
#                     (np.zeros(self.tau_max[0]+L), 
#                      np.zeros(self.tau_max[1]+L),
#                      np.zeros(self.tau_max[2]+L),
#                      np.zeros(self.tau_max[3]+L),
#                      np.zeros(self.tau_max[4]+L))
                    
#               }
}

def calc_initial_passenger_demand(arr_rate):

    res = np.zeros(R)
    denom = sum((arr_rate))
    for i in range(R-1):
        res[i] = np.floor(arr_rate[i] / denom * N)
    res[-1] = N - sum(res)
#     assert(sum(res) == N)

    return res

calc_initial_passenger_demand([1.8, 1.8, 1.8, 1.8, 18])
