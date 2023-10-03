#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from custom_envs.mpe import simple_spread_c_v2 
from algorithms.mappo.config import get_config
from algorithms.mappo.envs.mpe.MPE_env import MPEEnv
from algorithms.mappo.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
# 记住他用自己搞得环境弄完最后又搞了标准环境！

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = simple_spread_c_v2.parallel_env(N=all_args.num_agents, penalty_ratio=all_args.com_ratio,
                full_comm=all_args.full_comm, local_ratio=all_args.local_ratio, continuous_actions=True)
            #"--full_comm", action='store_true', help="if agents have full communication"
            #parser.add_argument("--com_ratio", type=float, default=0.5, help="Ratio for agent communication penalty")
            # parser.add_argument("--local_ratio", type=float, default=0.5, help="Ratio for agent rewards")
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        #By setting up multiple parallel environments, the training process can 
        # collect experiences from multiple episodes simultaneously, which 
        # can significantly speed up the data collection process.
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    # #     If n_rollout_threads is 1:
    #     DummyVecEnv is used, which is a simple wrapper that doesn't actually 
    #     parallelize the environment but keeps the same interface as the parallelized 
    #     version. This is suitable for scenarios where you don't need parallelization, 
    #     e.g., debugging or when running on a system with limited computational resources.
    #    If n_rollout_threads is greater than 1:
    #     SubprocVecEnv is used. This class creates multiple environments in separate
    #      processes. This allows for true parallelization where each environment runs
    #      in its own process, and experiences from each environment are collected 
    #      simultaneously. This is suitable for speeding up training by collecting more 
    #      data in the same amount of time.
    # simple_spread_c_v2.parallel_env通过这个构建训练环境，这里只有一个scenario,simple_spread.
    # 总的来说simple_spread_c.py有三个组成部分，scenario,world and components. world.core.py provide components,
    # scenario.py sets scenario,simple_env.py provides API like gym.


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # This function accepts two parameters: args, 
    # which is a list of command-line arguments (similar to sys.argv,type in terminal), 
    # and parser, which is an instance of the argparse.ArgumentParser class.
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]
    #     The parse_known_args method of the parser object is used to parse 
    # the arguments provided in args. It returns two values: the first is a 
    # namespace containing the parsed arguments, and the second is a list of 
    # all arguments that were not recognized.
    # By indexing with [0], the function retrieves only the namespace with 
    # the recognized arguments.

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.episode_length *= all_args.n_trajectories
    # refers to the number of episodes or trajectories collected by each parallel 
    # environment during 
    # a single sampling operation. 
    # 长度×个数，或许这是决定minibatch的参数
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    # The line torch.autograd.set_detect_anomaly(True, check_nan=True) is used to enable the
    # PyTorch autograd anomaly detection mode. This is mainly for debugging purposes 
    # to identify operations that result in "Not a Number" (NaN) or "Infinite" (Inf)
    #  values during the backward pass.
    # When this mode is turned on, PyTorch will perform extra checks 
    # at runtime, and if it detects that a NaN or Inf value is being
    #  produced during gradient calculation, it will throw an error and
    #  show the stack trace. This can help you pinpoint exactly where the
    #  problematic operation occurred.
    print(all_args.use_centralized_V)
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False

    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError
    # The term "recurrent policy" generally refers to the use of a recurrent neural network (RNN) 
    # When a policy is "recurrent," it means that the network has memory of past states, 
    # which can be important in partially observable environments where the current state
    #  does not contain all the information needed to make optimal decisions.
    # #########
    #########
    ######
    # it is a important part for hierarchical?
    # 主要还是环境部分可观测的原因

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/runs") / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # store resultos.path.abspath(__file__) gets the absolute path of the script 
    # that is being executed.
    # os.path.dirname() obtains the directory name from this absolute path.
    # os.path.split() splits this directory name into a head and a tail, where the 
    # tail is the last part of the path and the head is everything before it.
    # The [0] then selects the head part, essentially going up one directory level.
    # This path is then concatenated with "/runs", making a new directory where the 
    # runs will be stored.
    # Further subdirectories are then made for the specific algorithm (all_args.algorithm_
    # name) and the specific experiment (all_args.experiment_name).
    # wandb
    all_args.use_wandb = True
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    # title the process name
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from algorithms.mappo.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from algorithms.mappo.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

def simple_train(args):
    parser = get_config()
    all_args = parse_args("", parser)
    all_args.episode_length *= all_args.n_trajectories
    
    for key in args:
        setattr(all_args, key, args[key])

    if all_args.use_popart:
        all_args.use_valuenorm = False
    else:
        all_args.use_valuenorm = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/runs") / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    all_args.use_wandb = False
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from algorithms.mappo.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from algorithms.mappo.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])
# In essence, main is more comprehensive and geared towards being run directly from a
#  command line, with settings passed in as command-line arguments. simple_train seems 
# to be a more simplified or specialized version, possibly intended to be used 
# programmatically within other Python scripts. It is more flexible in terms of input
#  but skips some of the settings and checks present in main.
