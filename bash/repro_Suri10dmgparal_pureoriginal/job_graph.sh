#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/repro_Suri10dmgparal_pureoriginal/graph_mercury.out
#SBATCH --error=./job-outs/repro_Suri10dmgparal_pureoriginal/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration3jump.py --dataname  repro_Suri10dmgparal_pureoriginal --pdfname mercury --psi0arr 0.005 --psi1arr 0.5  --xiaarr  0.0002 0.0002 1000.   --xiparr  0.025 0.050 1000.     --hK 0.2 --hY 0.2 --hL 0.2 --Y_max_short 3.0

echo "Program ends $(date)"

