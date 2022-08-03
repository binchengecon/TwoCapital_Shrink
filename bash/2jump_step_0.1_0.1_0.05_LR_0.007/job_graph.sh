#! /bin/bash


######## login 
#SBATCH --job-name=graph2
#SBATCH --output=./job-outs/2jump_step_0.1_0.1_0.05_LR_0.007/graph_mercury.out
#SBATCH --error=./job-outs/2jump_step_0.1_0.1_0.05_LR_0.007/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_2jump.py --dataname  2jump_step_0.1_0.1_0.05_LR_0.007 --pdfname mercury --psi0arr 0.008 --psi1arr 0.8     --hXarr 0.1 0.1 0.05 --Xminarr 4.00 0.0 -5.5 0.0 --Xmaxarr 9.00 4.0 0.0 3.0

echo "Program ends $(date)"

