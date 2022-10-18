#! /bin/bash

# # action_name="repro_Suri10dmgparal_pureoriginal"
# # server_name="mercury"
# action_name="repro_Suri20dmgparal_pureoriginal2"
# server_name="mercury"



# xi_a=(0.0002 0.0002 1000.)
# xi_p=(0.025 0.050 1000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# psi0arr=(0.005)
# psi1arr=(0.5)

# # under "repro_Suri20dmgparal_pureoriginal2"
# NUM_DAMAGE=20
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# hK=0.2
# hY=0.2
# hL=0.2
# Y_max_short=3.0


action_name="rep_20dmgparal"
server_name="mercury"
# under "rep_20dmgparal"

hK=0.1
hY=0.1
hL=0.1
Y_max_short=3.0
xi_a=(10000.)
xi_p=(10000.)
psi0arr=(0.01)
psi1arr=(0.5)



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

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration3jump.py --dataname  $action_name --pdfname $server_name --psi0arr ${psi0arr[@]} --psi1arr ${psi1arr[@]}  --xiaarr  ${xi_a[@]}   --xiparr  ${xi_p[@]}     --hK $hK --hY $hY --hL $hL --Y_max_short $Y_max_short

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job_graph.sh

