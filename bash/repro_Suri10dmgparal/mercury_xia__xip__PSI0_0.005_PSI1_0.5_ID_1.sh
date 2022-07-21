#! /bin/bash


######## login 
#SBATCH --job-name=post_7
#SBATCH --output=./job-outs/repro_Suri10dmgparal/xia__xip__PSI0_0.005_PSI1_0.5/mercury_post_1.out
#SBATCH --error=./job-outs/repro_Suri10dmgparal/xia__xip__PSI0_0.005_PSI1_0.5/mercury_post_1.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/postdamage_spe_xi_psi_gammalist_name.py --num_gamma 10 --xi_a  --xi_g    --id 1 --psi_0 0.005 --psi_1 0.5 --name repro_Suri10dmgparal --hK 0.2 --hY  	0.2 --hL  0.2

echo "Program ends $(date)"

