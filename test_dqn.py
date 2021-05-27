
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from RH_PPO import *
from RH_DQN import *

import RideHailing
from RideHailing.envs import *
from RideHailing.envs.RideHailing_env import *
env = gym.make('RideHailing-v0', config=CONFIG)
# env = gym.make('CartPole-v1')

#print(env.state)
#print(env.action_space)
#print(env.observation_space) # openAI calles states observations :/
#print(env.action_space.sample())

#env.reset()
#tot_reward = 0

def evaluate(model, env, numiters):
    tot_reward_l = []

    for i in range(numiters):
        print('iteration'+str(i))
        env.reset()
        tot_reward = 0
        episode_over = False
        while (episode_over != True):
            _, reward, episode_over, _ = env.step(model(env.state))
            tot_reward += reward
        tot_reward_l.append(tot_reward)
    l = np.asarray(tot_reward_l)
    return (np.mean(l), np.std(l))


def random_model(state):
    # TODO: do not hardcode
    action = [state['origin'], np.random.randint(5)]
    return action

env = gym.make('RideHailing-v0', config=CONFIG)
# # why is the environment not resetting correctly????
# # import pdb; pdb.set_trace()

# result = evaluate(random_model, env, 1000)
# print(result)
# print(CONFIG)
perf_mean, perf_std = run_DQN(env,10,8)
print([perf_mean, perf_std])


# import pdb; pdb.set_trace()
# model.learn(total_timesteps=10000)

# model = evalua("MlpPolicy", env, verbose=1)
#result = evaluate(random_model, env, 1000)
#print(result)


#np.random.seed(0)
#cl = RideHailingEnv(CONFIG)
#for i in range(5):
#    for j in range(5):
#        print('State: {}'.format(cl.step([i, j])))
#    
#print('FINISHED')



