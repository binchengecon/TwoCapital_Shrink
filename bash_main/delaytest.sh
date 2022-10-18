#! /bin/bash


######## login 
#SBATCH --job-name=test
#SBATCH --output=./job-outs/compare/test2.out
#SBATCH --error=./job-outs/compare/test2.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

# main program


python3 ./abatement/postdamage_2jump_interp_SG.py

echo "Program ends $(date)"

