#! /bin/bash


######## login 
#SBATCH --job-name=repro_2
#SBATCH --output=./job-outs/repro_Suri10dmg/mercury_2.out
#SBATCH --error=./job-outs/repro_Suri10dmg/mercury_2.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/abatement.py --xi_p 1000.0 --xi_a 1000.0

echo "Program ends $(date)"

