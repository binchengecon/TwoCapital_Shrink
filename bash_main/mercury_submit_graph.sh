#! /bin/bash

action_name="replicateSuri"
server_name="mercury"


hK=0.1
hY=0.1
hL=0.2

Y_max_short=2.5
# Y_max_short=3.0

psi0arr=(0.008 0.010 0.012)
psi1arr=(0.8)


if [ -f ./bash/${action_name}/job_graph.sh ]
then
		rm ./bash/${action_name}/job_graph.sh
fi
mkdir -p ./bash/${action_name}/

touch ./bash/${action_name}/job_graph.sh


tee -a ./bash/${action_name}/job_graph.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/${action_name}/graph_${server_name}.out
#SBATCH --error=./job-outs/${action_name}/graph_${server_name}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration.py --dataname  $action_name --pdfname $server_name --psi0arr ${psi0arr[@]} --psi1arr ${psi1arr[@]} --hK $hK --hY $hY --hL $hL --Y_max_short $Y_max_short

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job_graph.sh

