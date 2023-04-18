from ray import air
import argparse
from ray import tune
from train_mappo import simple_train

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('num_agents', type=int, default=2)
  parser.add_argument('--logdir', type=str, default="/ray_results")
  parser.add_argument('--full_com', action='store_true')
  return parser.parse_args()

best_rewards = {}
iterations_since_best = {}

def stopper(trial_id, result):
    key = str(trial_id)
    global best_rewards, iterations_since_best

    if not key in best_rewards:
       best_rewards[key] = -float('inf')
       iterations_since_best[key] = 0
      
    improvement_threshold = 0.1
    patience = 1e6

    if result["average_episode_rewards"] > best_rewards[key]+ improvement_threshold:
        best_rewards[key] = result["average_episode_rewards"]
        iterations_since_best[key] = 0
    else:
        iterations_since_best[key] += 1

    if iterations_since_best[key] >= patience:
        return True
    else:
       return False

if __name__ == '__main__':
  args = parse_args()

  config = {
      "algorithm_name": tune.choice(["mappo", 'rmappo']),
      "env_name": "MPE",
      "scenario_name": "simple_spread",
      "experiment_name": "mappo_search_" + str(args.num_agents) + "_agents" + ("_full_com" if args.full_com else "_limited_com"),
      "cuda": False,
      "n_training_threads": 1,
      "n_rollout_threads": 4,
      "n_eval_rollout_threads": 1,
      "n_render_rollout_threads": 1,
      "num_env_steps": 1e7,
      "pop_art": tune.choice([True, False]),
      "episode_length": tune.choice([25, 50, 100]),
      "env_name": "MPE",
      "num_agents": args.num_agents,
      "share_policy": True,
      "use_centralized_V": True,
      "actor_hidden_size": tune.grid_search([32, 128, 256, 512, 1024, 2048]),
      "critic_hidden_size": tune.grid_search([32, 128, 256, 512, 1024, 2048]),
      "layer_N": tune.grid_search([1, 2, 3]),
      "use_ReLU": tune.choice([True, False]),
      "critic_lr": tune.uniform(1e-7, 1e-4),
      "ppo_epoch": tune.randint(1, 20),
      "clip_param": tune.uniform(0.1, 0.5),
      "gae_lambda":tune.uniform(0.9, 1),
      "gamma":tune.uniform(0.9, 1),
      "lr": tune.uniform(1e-7, 1e-4),
      "entropy_coef": tune.uniform(0.001, 0.1),
      "comm_penatly": tune.uniform(0.001, 1),
      "local_ratio": 0.5,
      "full_comm": args.full_com,
  }

  tune_config = tune.TuneConfig(
    mode="max",
    metric="average_episode_rewards",
    num_samples=10
  )
  run_config =air.RunConfig(
    local_dir=args.logdir,
    stop=stopper,
  )
  trainable_with_resources = tune.with_resources(simple_train, {"cpu": 4})
  analysis = tune.Tuner(
    trainable_with_resources,
    param_space=config,
    tune_config=tune_config,
    run_config=run_config,
  )
  analysis.fit()

