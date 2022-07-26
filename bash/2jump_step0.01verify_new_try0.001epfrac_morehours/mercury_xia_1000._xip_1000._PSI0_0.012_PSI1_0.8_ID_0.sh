#! /bin/bash


######## login 
#SBATCH --job-name=post17_2
#SBATCH --output=./job-outs/2jump_step0.01verify_new_try0.001epfrac_morehours/xia_1000._xip_1000._PSI0_0.012_PSI1_0.8/mercury_post_0.out
#SBATCH --error=./job-outs/2jump_step0.01verify_new_try0.001epfrac_morehours/xia_1000._xip_1000._PSI0_0.012_PSI1_0.8/mercury_post_0.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/postdamage_2jump.py --num_gamma 3 --xi_a 1000. --xi_g 1000.  --epsilonarr 0.1 0.001  --fractionarr 0.1 0.001   --maxiterarr 60000 80000  --id 0 --psi_0 0.012 --psi_1 0.8 --name 2jump_step0.01verify_new_try0.001epfrac_morehours --hXarr 0.01 0.01 0.01 --Xminarr 4.00 0.0 -5.5 0.0 --Xmaxarr 9.00 4.0 0.0 3.0

echo "Program ends $(date)"

