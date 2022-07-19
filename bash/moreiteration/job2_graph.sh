#! /bin/bash

######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/moreiteration/graph_midway.out
#SBATCH --error=./job-outs/moreiteration/graph_midway.err
#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

echo "$SLURM_JOB_NAME"

python /home/bincheng/TwoCapital_Bin/abatement/Result_spe_name_moreiteration.py --dataname  moreiteration --pdfname midway

echo "Program ends $(date)"

