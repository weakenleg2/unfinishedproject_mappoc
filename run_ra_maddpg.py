import os
import argparse
import torch
import numpy as np
from custom_envs import simple_spread_c_v2
from matplotlib import pyplot as plt
from algorithms.resource_aware_maddpg import RA_MADDPG
from utils.buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
update_counter = 0
writer = SummaryWriter()
model_path = 'models/'
figure_path = 'figures/'

def dict_to_tensor(d, unsqueeze_axis=0):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d).unsqueeze(unsqueeze_axis)
  return d


def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  # obs = torch.log(obs + 1e-4)
  # obs = obs - obs.mean()
  # obs = obs / (obs.std() + 1e-8)
  return obs


def get_actions(obs, env, agents, training=True):
  actions = {}
  logits = agents.step(obs, training)
  for i, agent in enumerate(env.possible_agents):
    actions[agent] = logits[i]
    #actions[agent][-1] = -1

  return actions, logits


def run_episode(env, agents, replay_buffer, training=True):
    global update_counter
    obs = env.reset()
    obs = preprocess_obs(obs)
    #agents.reset_noise()

    tot_reward = np.zeros(agents.n_agents)
    steps = 0
    tot_comms = 0

    while env.agents:
      actions, logits = get_actions(obs.to(device), env, agents, training)
      next_obs, rewards, dones, truncations, infos = env.step(actions)

      next_obs = preprocess_obs(next_obs)
      rewards = dict_to_tensor(rewards)
      dones = dict_to_tensor(dones)

      replay_buffer.push(obs, logits,
                         rewards, next_obs, dones)

      if len(replay_buffer) > 128 and training and update_counter > 100:
        update_counter = 0
        for j in range(agents.n_agents):
          sample = replay_buffer.sample(128, True, norm_rews=False)
          agents.update(sample, j, logger=writer)
        agents.update_all_targets()

      obs = next_obs
      update_counter += 1
      steps += 1
      tot_comms += infos['comms']
      tot_reward = tot_reward + rewards.squeeze().numpy()

    return tot_reward, tot_comms, steps

def plot_rewards(reward_history, comm_history):
  plt.plot(reward_history)
  plt.xlabel('Episode')
  plt.ylabel('Average reward per agent')
  plt.savefig(figure_path + 'reward_history.png')
  plt.close()

  plt.plot(comm_history)
  plt.xlabel('Episode')
  plt.ylabel('Communication-savings per episode')
  plt.savefig(figure_path + 'comm_history.png')
  plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--load', type=str)
  parser.add_argument('-n', '--n_agents', type=int)
  args = parser.parse_args()

  n_agents = 3

  if args.n_agents:
    n_agents = args.n_agents

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(figure_path):
    os.makedirs(figure_path)

  env = simple_spread_c_v2.parallel_env(N=n_agents, communication_penalty=-0.01,
                                        local_ratio=0.5, max_cycles=40, continuous_actions=True)

  obs_dim = env.observation_space('agent_0').shape[0]
  action_dim = env.action_space('agent_0').shape[0]
  replay_buffer = ReplayBuffer(int(1e6), num_agents=n_agents,
                               obs_dims=[obs_dim for _ in range(n_agents)],
                               ac_dims=[action_dim for _ in range(n_agents)])

  if args.load:
    algo = RA_MADDPG.init_from_save(args.load, device='cuda')
  else:
    algo = RA_MADDPG(in_dim=obs_dim, out_dim=action_dim, 
                   n_agents=n_agents,
                   hidden_dim=256,
                   discrete_action=False,
                   device='cuda',
                   gamma=0.9, lr=1e-4, tau=5e-2)

  e_max = 50000
  best = -1000000000
  for i in range(e_max):
    tot_reward, comms, steps = run_episode(env, algo, replay_buffer, training=True)

    comm_savings = 1 - (comms / (steps * n_agents))
    avg_reward = sum(tot_reward)/n_agents
    writer.add_scalar('agent/reward', avg_reward, i)
    writer.add_scalar('agent/comm_savings', comm_savings, i)

    if i % 100 == 0:
      eval_tot_reward = 0
      eval_tot_comms = 0
      eval_steps = 0

      for _ in range(5):
        rewards, comms, steps = run_episode(env, algo, replay_buffer, training=False)

        eval_steps += steps
        eval_tot_comms += comms
        eval_tot_reward += sum(rewards) / n_agents

      comm_ratio = eval_tot_comms / (eval_steps * n_agents)
      comm_ratio /= 5

      eval_comm_savings = 1 - comm_ratio
      eval_comm_savings /= 5
      eval_tot_reward /= 5

      # reward_history.append(eval_tot_reward)
      print('------------------------------------')
      print('Episode: ' + str(i) + '/' + str(e_max))
      print('Avg reward: ' + str(eval_tot_reward))
      print('Number of communications: ' + str(eval_tot_comms) + '/' + str(eval_steps * n_agents))

    if eval_tot_reward >= best:
      algo.save(model_path +'best.pt')
      best = eval_tot_reward

    if i % 1000 == 0:
      algo.save(model_path + str(i) + '.pt')

  writer.close()
  env.close()
