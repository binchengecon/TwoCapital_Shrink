#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/rep_20dmgparal/graph_mercury.out
#SBATCH --error=./job-outs/rep_20dmgparal/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration3jump.py --dataname  rep_20dmgparal --pdfname mercury --psi0arr 0.01 --psi1arr 0.5  --xiaarr  10000.   --xiparr  10000.     --hK 0.1 --hY 0.1 --hL 0.1 --Y_max_short 3.0

echo "Program ends $(date)"

