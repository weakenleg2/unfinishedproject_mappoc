import time
import argparse
import gym
from pettingzoo.mpe import simple_reference_v2
import numpy as np
import torch
from algorithms.maddpg import MADDPG

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
  logits = agents.step(obs, training)[0].detach()
  action_list = torch.argmax(logits, dim=1).tolist()
  for i, agent in enumerate(env.possible_agents):
    actions[agent] = action_list[i]

  return actions, logits

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=str)
  args = parser.parse_args()

  n_agents = 2

  maddpg = MADDPG.init_from_save(args.filename)
  maddpg.move_to_device(device='cpu')

  env = simple_reference_v2.parallel_env( 
                                      local_ratio = 0.5, 
                                      max_cycles=120, 
                                      continuous_actions=False,
                                      render_mode = 'human')


  obs = env.reset()
  obs = preprocess_obs(obs)
  tot_reward = np.zeros(n_agents)

  while env.agents:
    actions, logits = get_actions(obs, env, maddpg, False)
    next_obs, rewards, dones, truncations, infos = env.step(actions)
    next_obs = preprocess_obs(next_obs)
    obs = next_obs
    rewards = dict_to_tensor(rewards)
    env.render()
    time.sleep(0.03)
    tot_reward = tot_reward + rewards.squeeze().numpy()

  print(tot_reward)
  env.close()
