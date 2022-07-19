#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/comparisonwitholdpsi01/graph_mercury.out
#SBATCH --error=./job-outs/comparisonwitholdpsi01/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name_moreiteration.py --dataname  comparisonwitholdpsi01 --pdfname mercury --psi0arr 0.005 --psi1arr 0.5 --hK 0.2 --hY 0.2 --hL 0.2

echo "Program ends $(date)"

