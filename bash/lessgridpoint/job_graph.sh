#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/lessgridpoint/graph_mercury.out
#SBATCH --error=./job-outs/lessgridpoint/graph_mercury.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "$SLURM_JOB_NAME"
echo "Program starts $(date)"

python3 /home/bcheng4/TwoCapital_Bin/abatement/Result_spe_name_moreiteration.py --dataname  lessgridpoint --pdfname mercury

echo "Program ends $(date)"

