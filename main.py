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
                               obs_dims=[obs_space[0] for _ in range(4)], 
                               ac_dims=[2, 2, 2, 2])

  critic_obs_space = 4 * (obs_space[0] + env.action_space[0].n)
  #print(critic_obs_space)
  for _ in range(4):
    config.append({
      "num_in_pol": obs_space[0],
      "num_out_pol": 2,
      "num_in_critic": critic_obs_space,
    })

  maddpg = MADDPG(config, 4, discrete_action=True, device='cpu')
  maddpg.prep_training(device='cpu')


  for i in range(100):
    obs = (env.reset())
    obs = np.array(obs)
    obs = torch.tensor(obs)
    obs = torch.unsqueeze(obs, 0)

    dones = torch.tensor([False, False, False, False], dtype=bool)
    tot_reward = 0
    while not dones.all().item():
      actions = maddpg.step(obs)[0]
      actions = np.array(actions)
      actions = torch.tensor(actions)

      #Oklart om denna ska skickas in som target för nn agent sen. 
      # Eller om man ska ha one-hot
      # Prova 
      actions = torch.argmax(actions, dim=1)

      next_obs, rewards, dones, infos = env.step(actions.tolist())

      rewards = torch.tensor(rewards).unsqueeze(0)
      next_obs = np.array(next_obs)
      next_obs = torch.tensor(next_obs).unsqueeze(0)
      dones = torch.tensor(dones).unsqueeze(0)

      replay_buffer.push(obs, actions, rewards, next_obs, dones)
      tot_reward += sum(rewards.squeeze()).item()

      if len(replay_buffer) > 1000:
        for j in range(4):
          sample = replay_buffer.sample(1000)
          maddpg.update(sample, j)

        maddpg.update_all_targets()

      #time.sleep(0.5)
      #env.render()
      obs = next_obs

    print(tot_reward)

  maddpg.save('Model.memes')
  env.close()

