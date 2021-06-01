#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=conformer-rl-test
#SBATCH --mail-user=runxuanj@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env

# The application(s) to execute along with its input arguments and options:
conda activate my-rdkit-env
python -u testa2c.py