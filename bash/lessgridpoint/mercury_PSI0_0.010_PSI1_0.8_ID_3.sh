#! /bin/bash


######## login 
#SBATCH --job-name=post_10
#SBATCH --output=./job-outs/lessgridpoint/0.010_0.8/mercury_post_3.out
#SBATCH --error=./job-outs/lessgridpoint/0.010_0.8/mercury_post_3.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --id 3 --psi_0 0.010 --psi_1 0.8 --name lessgridpoint

echo "Program ends $(date)"

