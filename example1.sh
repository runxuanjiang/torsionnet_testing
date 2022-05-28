#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example1
#SBATCH --mail-user=runxuanj@umich.edu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --account=tewaria0
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env

# The application(s) to execute along with its input arguments and options:
source /home/${USER}/.bashrc
module load gcc/9.2.0
conda activate rl-pip
python -u example1.py