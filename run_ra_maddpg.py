import time
import os
import argparse
import torch
import numpy as np
import gym
from pettingzoo.mpe import simple_reference_v2
from matplotlib import pyplot as plt
from algorithms.resource_aware_maddpg import RA_MADDPG
# from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
update_counter = 0


def dict_to_tensor(d, unsqueeze_axis=0):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d).unsqueeze(unsqueeze_axis)
  return d


def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  # obs = obs - obs.mean()
  # obs = obs / (obs.std() + 1e-8)
  return obs


def get_actions(obs, env, agents, training=True):
  actions = {}
  options, log= agents.step(obs, training)
  print(actions)
  action_list = torch.argmax(actions[0], dim=-1).tolist()
  for i, agent in enumerate(env.possible_agents):
    actions[agent] = action_list[i]

  return actions, logits


def run_episode(env, agents, replay_buffer, training=True):
    global update_counter
    obs = env.reset()
    obs = preprocess_obs(obs)

    tot_reward = np.zeros(agents.n_agents)

    while env.agents:
      actions, logits = get_actions(obs.to(device), env, agents, training)
      next_obs, rewards, dones, truncations, infos = env.step(actions)

      next_obs = preprocess_obs(next_obs)
      rewards = dict_to_tensor(rewards)
      dones = dict_to_tensor(dones)

      replay_buffer.push(obs, logits,
                         rewards, next_obs, dones)

      if len(replay_buffer) > 1024 and training and update_counter > 20:
        update_counter = 0
        for j in range(agents.n_agents):
          sample = replay_buffer.sample(1024, True)
          maddpg.update(sample, j)
        maddpg.update_all_targets()

      obs = next_obs
      update_counter += 1
      tot_reward = tot_reward + rewards.squeeze().numpy()

    return tot_reward


def plot_rewards(reward_history):
  plt.plot(reward_history)
  plt.legend(['Agent 1', 'Agent 2', 'Agent 3'])
  plt.xlabel('Episode')
  plt.ylabel('Total reward')
  plt.savefig('reward_history.png')
  plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--load', type=str)
  args = parser.parse_args()
  n_agents = 2

  if not os.path.exists('models'):
    os.makedirs('models')

  env = simple_reference_v2.parallel_env(
      local_ratio=0.5, max_cycles=25, continuous_actions=True)
  print(env.action_space('agent_0'))
  quit()

  obs_dim = env.observation_space('agent_0').shape[0]
  replay_buffer = ReplayBuffer(int(1e6), num_agents=n_agents,
                               obs_dims=[obs_dim for _ in range(n_agents)],
                               ac_dims=[env.action_space('agent_0').n for _ in range(n_agents)])

  critic_obs_space = n_agents * (obs_dim + env.action_space('agent_0').n + 2)

  config = []
  for _ in range(n_agents):
    config.append({
        "in_dim": obs_dim,
        "out_dim": env.action_space('agent_0').n,
        "critic_in": critic_obs_space,
    })

  if args.load:
    maddpg = RA_MADDPG.init_from_save(args.load)
  else:
    maddpg = RA_MADDPG(config, n_agents, hidden_dim=64,
                    discrete_action=False, device='cuda', gamma=0.95, lr=1e-2, tau=1e-2)

  maddpg.move_to_device(training=True, device='cuda')

  e_max = 25000
  reward_history = []
  best = -1000000000
  for i in range(e_max):
    tot_reward = run_episode(env, maddpg, replay_buffer, training=True)
    reward_history.append(tot_reward)

    if i % 100 == 0:
      eval_tot_reward = 0
      for _ in range(5):
        rewards = run_episode(env, maddpg, replay_buffer, training=False)
        eval_tot_reward += sum(rewards)

      eval_tot_reward /= 5

      # reward_history.append(eval_tot_reward)
      print('------------------------------------')
      print('Episode: ' + str(i) + '/' + str(e_max))
      print('Avg reward: ' + str(eval_tot_reward))

      plot_rewards(reward_history)
    if eval_tot_reward >= best:
      maddpg.save('models/best.pt')

      best = eval_tot_reward
    if i % 1000 == 0:
      maddpg.save('models/' + str(i) + '.pt')

  env.close()
