import ray
from ray import air
import pathlib
import argparse
from ray import tune
from train_mappo import simple_train

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('num_agents', type=int, default=2)
  parser.add_argument('--logdir', type=str, default="/ray_results")
  parser.add_argument('--full_com', action='store_true')
  return parser.parse_args()

def stopper(trial_id, result):
  return result["average_episode_rewards"] < -3

if __name__ == '__main__':
  args = parse_args()

  config = {
      "algorithm_name": tune.choice(["mappo", 'rmappo']),
      "env_name": "MPE",
      "scenario_name": "simple_spread",
      "experiment_name": "mappo_search_" + str(args.num_agents) + "_agents",
      "cuda": False,
      "n_training_threads": 1,
      "n_rollout_threads": 1,
      "n_eval_rollout_threads": 1,
      "n_render_rollout_threads": 1,
      "num_env_steps": 1e6,
      "pop_art": tune.choice([True, False]),
      "episode_length": tune.choice([25, 50, 100]),
      "env_name": "MPE",
      "num_agents": args.num_agents,
      "share_policy": True,
      "use_centralized_V": True,
      "actor_hidden_size": tune.grid_search([32, 128, 256, 512, 1024]),
      "critic_hidden_size": tune.grid_search([32, 128, 256, 512, 1024]),
      "layer_N": tune.grid_search([1, 2, 3]),
      "use_ReLU": tune.choice([True, False]),
      "critic_lr": tune.uniform(1e-7, 1e-2),
      "ppo_epoch": tune.randint(1, 20),
      "clip_param": 0.2,
      "gae_lambda":tune.uniform(0.9, 1),
      "gamma":tune.uniform(0.9, 1),
      "lr": tune.uniform(1e-7, 1e-4),
      "entropy_coef": tune.uniform(0.001, 0.1),
      "comm_penatly": tune.uniform(0.001, 1),
      "critic_lr": tune.uniform(1e-7, 1e-4),
      "local_ratio": tune.uniform(0.1, 0.9),
      "full_comm": args.full_com,
  }

  tune_config = tune.TuneConfig(
    mode="max",
    metric="average_episode_rewards",
  )
  run_config =air.RunConfig(
    local_dir=args.logdir,
    stop=stopper,
  )
  analysis = tune.Tuner(
    simple_train,
    param_space=config,
    tune_config=tune_config,
    run_config=run_config,
  )
  analysis.fit()

  #main(config, cmd_line=False)
