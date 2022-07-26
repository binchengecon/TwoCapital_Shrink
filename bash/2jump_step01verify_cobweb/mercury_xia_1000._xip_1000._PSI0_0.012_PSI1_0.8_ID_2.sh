#! /bin/bash


######## login 
#SBATCH --job-name=post11_8
#SBATCH --output=./job-outs/2jump_step01verify_cobweb/xia_1000._xip_1000._PSI0_0.012_PSI1_0.8/mercury_post_2.out
#SBATCH --error=./job-outs/2jump_step01verify_cobweb/xia_1000._xip_1000._PSI0_0.012_PSI1_0.8/mercury_post_2.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/postdamage_2jump_cobweb.py --num_gamma 3 --xi_a 1000. --xi_g 1000.  --epsilonarr 0.1 0.01  --fractionarr 0.1 0.01   --maxiterarr 40000 60000  --id 2 --psi_0 0.012 --psi_1 0.8 --name 2jump_step01verify_cobweb --hXarr 0.1 0.1 0.1 --Xminarr 4.00 0.0 -5.5 0.0 --Xmaxarr 9.00 4.0 0.0 3.0

echo "Program ends $(date)"

