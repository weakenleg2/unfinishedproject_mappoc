import time
import argparse
import gym
from custom_envs.mpe import simple_spread_c_v2
import numpy as np
import torch
from mappo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from algorithms.resource_aware_maddpg import RA_MADDPG

def dict_to_tensor(d, unsqueeze_axis=0):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d)
  return d

def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  #obs = obs - obs.mean()
  # obs = obs / (obs.std() + 1e-8)
  return obs

def get_actions(obs, env, policy, training=False):
  actions = {}
  logits, _, _ = policy(obs, torch.tensor(0), torch.tensor(0))
  for i, agent in enumerate(env.possible_agents):
    #actions[agent] = torch.tensor([-1, 1, 0])
    actions[agent] = logits[i]

  return actions, logits

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=str)
  parser.add_argument('-n', '--n_agents', type=int, default=3)
  parser.add_argument('-r', '--random_actions', action='store_true')
  args = parser.parse_args()

  if not hasattr(args, 'hidden_size'):
    args.hidden_size = 256
    args.layer_N = 2
    args.gain = 0.01
    args.use_orthogonal = False
    args.use_policy_active_masks = True
    args.use_naive_recurrent_policy = False
    args.use_recurrent_policy = False
    args.recurrent_N = 0
    args.use_feature_normalization = True
    args.use_ReLU = False 
    args.stacked_frames= 1
    args.use_stacked_frames = False
    args.use_centralized_V = True
    args.share_policy = True
    args.pop_art = False 

  env = simple_spread_c_v2.parallel_env( 
                                      N=args.n_agents,
                                      local_ratio = 0.5, 
                                      max_cycles=25, 
                                      continuous_actions=True,
                                      render_mode = 'human')




  if not args.random_actions:
    state_dict = torch.load(args.filename)
    policy = R_Actor(args, obs_space=env.observation_space('agent_0'), action_space=env.action_space('agent_0'))
    policy.load_state_dict(state_dict)

  tot_reward = 0
  seeds = range(100)

  for s in seeds:
    seed_reward = np.zeros(args.n_agents)
    obs = env.reset()
    obs = preprocess_obs(obs)

    np.random.seed(s)
    torch.manual_seed(s)
    while env.agents:
      if args.random_actions:
        actions = {}
        for n in range(args.n_agents):
          a = env.action_space('agent_0').sample()
          a = (*a[0], a[1])
          actions['agent_{}'.format(n)] = a
        #print(actions)
      else:
        actions, logits = get_actions(obs, env, policy, False)

      #print(actions)
      next_obs, rewards, dones, truncations, infos = env.step(actions)
      next_obs = preprocess_obs(next_obs)
      obs = next_obs
      rewards = dict_to_tensor(rewards)
      #print(env.aec_env.env.scenario.n_collisions / 2)
      time.sleep(0.05)
      seed_reward = seed_reward + rewards.squeeze().numpy()
    tot_reward += seed_reward.mean()

  print(tot_reward / len(seeds))

  env.close()
