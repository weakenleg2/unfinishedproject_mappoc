#!/bin/sh
#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p node 
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -J "ra_mappo"
#SBATCH -o /home/pagliaro/project/RAC/outs/%j.out

PYTHON=/proj/uppmax2023-2-14/envs/RAC/bin/python3

env="MPE"
scenario="simple_spread" 
algo="mappo" #"mappo" 
seed=1

module load conda
conda activate RAC
cd /home/pagliaro/project/RAC

$PYTHON scripts/train_mappo.py --env_name ${env} --algorithm_name ${algo} \
    --scenario_name ${scenario} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 --lr 7e-4 --critic_lr 7e-4 $@
