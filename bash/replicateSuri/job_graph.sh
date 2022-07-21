#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/replicateSuri/graph_mercury.out
#SBATCH --error=./job-outs/replicateSuri/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration.py --dataname  replicateSuri --pdfname mercury --psi0arr 0.008 0.010 0.012 --psi1arr 0.8 --hK 0.1 --hY 0.1 --hL 0.2 --Y_max_short 2.5

echo "Program ends $(date)"

