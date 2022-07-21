#! /bin/bash


######## login 
#SBATCH --job-name=repro_0
#SBATCH --output=./job-outs/repro_Suri10dmg/mercury_0.out
#SBATCH --error=./job-outs/repro_Suri10dmg/mercury_0.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/abatement.py --xi_p 0.025 --xi_a 0.0002

echo "Program ends $(date)"

