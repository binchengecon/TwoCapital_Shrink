#! /bin/bash

######## login
#SBATCH --job-name=test
#SBATCH --output=./job-outs/upload.out
#SBATCH --error=./job-outs/upload.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5 gcc/9.2.0

# pip3 install dropbox.client

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"
start_time=$(date +%s)
# perform a task

python3 /home/bcheng4/TwoCapital_Shrink/abatement/UploadDB1.py

echo "Program ends $(date)"
end_time=$(date +%s)

# elapsed time with second resolution
elapsed=$((end_time - start_time))

eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
