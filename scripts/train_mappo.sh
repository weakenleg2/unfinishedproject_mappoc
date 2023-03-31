#!/bin/sh
#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -J "ra_mappo"
#SBATCH -o /home/pagliaro/project/RAC/outs/slurm.%j.out

env="MPE"
scenario="simple_spread" 
num_agents=3
algo="mappo" #"mappo" 
seed=1
exp="mappo_hidden_size_comparision"
hidden_sizes=(32 64 128 256)

#
#echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for size in ${hidden_sizes[@]};
do
    python train_mappo.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 40 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4  --hidden_size ${size}
done