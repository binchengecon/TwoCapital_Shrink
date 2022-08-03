#! /bin/bash


######## login 
#SBATCH --job-name=pre10_0
#SBATCH --output=./job-outs/2jump_step_0.1_0.1_0.05_LR_0.007/xia_1000._xip_1000._PSI0_0.008_PSI1_0.8/mercury_pre.out
#SBATCH --error=./job-outs/2jump_step_0.1_0.1_0.05_LR_0.007/xia_1000._xip_1000._PSI0_0.008_PSI1_0.8/mercury_pre.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/predamage_2jump.py --num_gamma 3 --xi_a 1000. --xi_p 1000.  --epsilonarr 0.005 0.005  --fractionarr 0.005 0.005   --maxiterarr 90000 90000  --psi_0 0.008 --psi_1 0.8 --name 2jump_step_0.1_0.1_0.05_LR_0.007 --hXarr 0.1 0.1 0.05 --Xminarr 4.00 0.0 -5.5 0.0 --Xmaxarr 9.00 4.0 0.0 3.0

echo "Program ends $(date)"

