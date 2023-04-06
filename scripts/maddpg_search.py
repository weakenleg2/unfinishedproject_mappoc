import ray
import pathlib
import argparse
from ray import tune
from train_mappo import simple_train

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('num_agents', type=int, default=2)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  config = {
      "scenario_name": "simple_spread",
      "experiment_name": "mappo_search_" + str(args.num_agents) + "_agents",
      "cuda": False,
      "seed": tune.randint(1,10),
      "n_training_threads": 1,
      "n_rollout_threads": 1,
      "n_eval_rollout_threads": 1,
      "n_render_rollout_threads": 1,
      "num_env_steps": 1e6,
      "pop_art": tune.choice([True, False]),
      "episode_length": tune.randint(25, 1000),
      "env_name": "MPE",
      "num_agents": args.num_agents,
      "share_policy": True,
      "use_centralized_V": True,
      "hidden_size": tune.grid_search([32, 128, 256, 512, 1024]),
      "layer_N": tune.grid_search([1, 2, 3]),
      "use_ReLU": tune.choice([True, False]),
      "critic_lr": tune.uniform(1e-7, 1e-2),
      "ppo_epoch": tune.randint(1, 20),
      "clip_param": 0.2,
      "gae_lambda":tune.uniform(0.9, 1),
      "gamma":tune.uniform(0.9, 1),
      "lr": tune.uniform(1e-7, 1e-2),
  }

  tune_config = tune.TuneConfig(
    mode="max",
    metric="average_episode_rewards",
  )
  analysis = tune.Tuner(
    simple_train,
    param_space=config,
    tune_config=tune_config,
  )
  analysis.fit()

  #main(config, cmd_line=False)
