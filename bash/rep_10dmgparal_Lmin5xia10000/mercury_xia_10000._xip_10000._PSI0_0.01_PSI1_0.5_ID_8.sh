#! /bin/bash


######## login 
#SBATCH --job-name=1e4post_8
#SBATCH --output=./job-outs/rep_10dmgparal_Lmin5xia10000/xia_10000._xip_10000._PSI0_0.01_PSI1_0.5/mercury_post_8.out
#SBATCH --error=./job-outs/rep_10dmgparal_Lmin5xia10000/xia_10000._xip_10000._PSI0_0.01_PSI1_0.5/mercury_post_8.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/postdamage_spe_xi_psi_gammalist_name3.py --num_gamma 10 --xi_a 10000. --xi_g 10000.  --epsilonarr 0.1 0.1 0.1  --fractionarr 0.1 0.01 0.01   --maxiterarr 30000 30000 30000  --id 8 --psi_0 0.01 --psi_1 0.5 --name rep_10dmgparal_Lmin5xia10000 --hK 0.1 --hY  	0.1 --hL  0.1

echo "Program ends $(date)"

