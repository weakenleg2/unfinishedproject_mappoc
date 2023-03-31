#!/bin/bash -l

#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -J "conda_install"

module load conda
conda activate RAC
conda install --name RAC pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
