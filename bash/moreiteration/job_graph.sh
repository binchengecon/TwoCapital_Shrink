#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/moreiteration/graph_mercury_newtimespan.out
#SBATCH --error=./job-outs/moreiteration/graph_mercury_newtimespan.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name_moreiteration.py --dataname  moreiteration --pdfname mercury_newtimespan --psi0arr 0.008 0.010 0.012 --psi1arr 0.8 --hK 0.1 --hY 0.1 --hL 0.1

echo "Program ends $(date)"

