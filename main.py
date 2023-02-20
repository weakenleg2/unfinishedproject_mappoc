import argparse
import time
import torch
import numpy as np
import gym
from algorithms.maddpg import MADDPG
#from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()

def run():
  pass

if __name__ == '__main__':
  config = []

  env = gym.make('ma_gym:TrafficJunction4-v0')
  obs_space = env.observation_space[0].shape
  replay_buffer = ReplayBuffer(10000, num_agents=4, 
                               obs_dims=obs_space, 
                               ac_dims=[2, 2, 2, 2])

  for _ in range(4):
    config.append({
      "num_in_pol": obs_space[0],
      "num_out_pol": 2,
      "num_in_critic": obs_space[0],
    })

  maddpg = MADDPG(config, 4, discrete_action=True)
  obs = (env.reset())
  obs = np.array(obs)
  obs = torch.tensor(obs)
  obs = torch.unsqueeze(obs, 0)

  for i in range(5):

    actions = maddpg.step(obs)[0]
    actions = np.array(actions)
    actions = torch.tensor(actions)

    actions = torch.argmax(actions, dim=1).tolist()
    print(actions)

    obs, _, _, _ = env.step(actions)

    obs = np.array(obs)
    obs = torch.tensor(obs)
    obs = torch.unsqueeze(obs, 0)

    env.render()
    time.sleep(1)
