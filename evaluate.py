import time
import argparse
import gym
from custom_envs import simple_spread_c_v2
import numpy as np
import torch
from algorithms.resource_aware_maddpg import RA_MADDPG

def dict_to_tensor(d, unsqueeze_axis=0):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d).unsqueeze(unsqueeze_axis)
  return d

def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  #obs = obs - obs.mean()
  # obs = obs / (obs.std() + 1e-8)
  return obs

def get_actions(obs, env, agents, training=True):
  actions = {}
  logits = agents.step(obs, training)
  for i, agent in enumerate(env.possible_agents):
    #actions[agent] = torch.tensor([-1, 1, 0])
    actions[agent] = logits[i]

  return actions, logits

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=str)
  parser.add_argument('-n', '--n_agents', type=int, default=3)
  args = parser.parse_args()

  maddpg = RA_MADDPG.init_from_save(args.filename, device='cpu')

  env = simple_spread_c_v2.parallel_env( 
                                      N=args.n_agents,
                                      local_ratio = 0.5, 
                                      max_cycles=25, 
                                      continuous_actions=True,
                                      render_mode = 'human')


  obs = env.reset()
  obs = preprocess_obs(obs)
  tot_reward = np.zeros(args.n_agents)

  while env.agents:
    actions, logits = get_actions(obs, env, maddpg, False)
    print(actions)
    next_obs, rewards, dones, truncations, infos = env.step(actions)
    next_obs = preprocess_obs(next_obs)
    obs = next_obs
    rewards = dict_to_tensor(rewards)
    env.render()
    time.sleep(0.03)
    tot_reward = tot_reward + rewards.squeeze().numpy()

  print(tot_reward)
  env.close()
