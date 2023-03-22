import os
import argparse
import torch
import numpy as np
from custom_envs import simple_spread_c_v2
from matplotlib import pyplot as plt
from algorithms.resource_aware_maddpg import RA_MADDPG
from utils.buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

USE_CUDA = False

device = torch.device("cuda" if USE_CUDA else "cpu")
update_counter = 0
writer = SummaryWriter()


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


def get_actions(obs, env, agents, full_comm=False, training=True):
  actions = {}
  logits = agents.step(obs, training)
  for i, agent in enumerate(env.possible_agents):
    actions[agent] = logits[i]
    if full_comm:
      actions[agent][-2] = 1
      actions[agent][-1] = 0

  return actions, logits


def run_episode(env, agents, replay_buffer, args, training=True):
    global update_counter
    obs = env.reset()
    obs = preprocess_obs(obs)

    tot_reward = np.zeros(agents.n_agents)
    steps = 0
    tot_comms = 0

    while env.agents:
      actions, logits = get_actions(obs.to(device), env, agents,
                                    full_comm=args.full_comm, training=training)
      next_obs, rewards, dones, _, infos = env.step(actions)

      next_obs = preprocess_obs(next_obs)
      rewards = dict_to_tensor(rewards)
      dones = dict_to_tensor(dones)

      replay_buffer.push(obs, logits,
                         rewards, next_obs, dones)

      if len(replay_buffer) > args.batch_size and training and update_counter > args.update_interval:
        update_counter = 0
        for j in range(agents.n_agents):
          sample = replay_buffer.sample(
              args.batch_size, USE_CUDA, norm_rews=False)
          agents.update(sample, j, logger=writer)
        agents.update_all_targets()

      obs = next_obs
      update_counter += 1
      steps += 1
      tot_comms += infos['comms']
      tot_reward = tot_reward + rewards.squeeze().numpy()

    return tot_reward, tot_comms, steps


'''def plot_rewards(argsmreward_history, comm_history):
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
'''


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--load', type=str)
  parser.add_argument('-n', '--n_agents', type=int, default=3)
  parser.add_argument('-b', '--buffer_size', type=int, default=1e6)

  parser.add_argument('--model_path', type=str, default="models/")
  parser.add_argument('--figure_path', type=str, default="figures/")
  parser.add_argument('--n_episodes', type=int, default=50000)
  parser.add_argument('--eval_episodes', type=int, default=10)
  parser.add_argument('--eval_interval', type=int, default=100)
  parser.add_argument('--save_interval', type=int, default=1000)
  parser.add_argument('--full_comm', type=bool, default=False)

  #Agent config
  parser.add_argument('--lr', type=float, default=1e-2)
  parser.add_argument('--tau', type=float, default=5e-2)
  parser.add_argument('-e', '--epsilon', type=float, default=1)
  parser.add_argument('--epsilon_decay', type=float, default=0.995)
  parser.add_argument('--gamma', type=float, default=0.95)
  parser.add_argument('--hidden_dim', type=int, default=64)

  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--update_interval', type=int, default=100)

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  n_agents = args.n_agents

  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

  if not os.path.exists(args.figure_path):
    os.makedirs(args.figure_path)

  env = simple_spread_c_v2.parallel_env(N=n_agents, communication_penalty=-0.1,
                                        local_ratio=0.5, max_cycles=40, continuous_actions=True)

  obs_dim = env.observation_space('agent_0').shape[0]
  action_dim = env.action_space('agent_0').shape[0]

  replay_buffer = ReplayBuffer(int(args.buffer_size), num_agents=n_agents,
                               obs_dims=[obs_dim for _ in range(n_agents)],
                               ac_dims=[action_dim for _ in range(n_agents)])

  if args.load:
    algo = RA_MADDPG.init_from_save(args.load, device=device)
  else:
    algo = RA_MADDPG(in_dim=obs_dim, out_dim=action_dim,
                     n_agents=n_agents,
                     hidden_dim=args.hidden_dim,
                     discrete_action=False,
                     eps=args.epsilon, 
                     eps_decay=args.epsilon_decay,
                     device=device,
                     gamma=args.gamma, lr=args.lr, tau=args.tau)

  best = -1000000000
  for i in range(args.n_episodes):
    tot_reward, comms, steps = run_episode(
        env, algo, replay_buffer, args, training=True)

    comm_savings = 1 - (comms / (steps * n_agents))
    avg_reward = sum(tot_reward)/n_agents

    writer.add_scalar('agent/reward', avg_reward, i)
    writer.add_scalar('agent/comm_savings', comm_savings, i)

    if i % args.eval_interval == 0:
      eval_tot_reward = 0
      eval_tot_comms = 0
      eval_steps = 0

      for _ in range(args.eval_episodes):
        rewards, comms, steps = run_episode(
            env, algo, replay_buffer, args, training=False)

        eval_steps += steps
        eval_tot_comms += comms
        eval_tot_reward += sum(rewards) / n_agents

      eval_tot_reward /= args.eval_episodes

      # reward_history.append(eval_tot_reward)
      print('------------------------------------')
      print('Episode: ' + str(i) + '/' + str(args.n_episodes))
      print('Avg reward: ' + str(eval_tot_reward))
      print('Number of communications: ' + str(eval_tot_comms) +
            '/' + str(eval_steps * n_agents))

    if eval_tot_reward >= best:
      algo.save(args.model_path + 'best.pt')
      best = eval_tot_reward

    if i % args.save_interval == 0:
      algo.save(args.model_path + str(i) + '.pt')

  writer.close()
  env.close()
