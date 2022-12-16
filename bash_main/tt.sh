#! /bin/bash

######## login
#SBATCH --job-name=post1_3937
#SBATCH --output=./job-outs/2jump_step_0.2_0.2_0.2_LR_0.008_Psi01ComparisonSlide/Bin.out
#SBATCH --error=./job-outs/2jump_step_0.2_0.2_0.2_LR_0.008_Psi01ComparisonSlide/Bin.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5 gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/abatement_Bin.py

echo "Program ends $(date)"
