#! /bin/bash


######## login 
#SBATCH --job-name=delay1
#SBATCH --output=./job-outs/checkconvergence/delaytest.out
#SBATCH --error=./job-outs/checkconvergence/delaytest.err

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


python3 ./abatement/checkSuriConvergenceLevel.py

echo "Program ends $(date)"

