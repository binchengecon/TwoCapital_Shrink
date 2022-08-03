#! /bin/bash


# actiontime=1
# action_name="2jump_beforemayversion"
# server_name="mercury"
# python_name="Result_2jump.py"


# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)


actiontime=2
action_name="2jump_step_0.1_0.1_0.05_LR_0.007"
server_name="mercury"
python_name="Result_2jump.py"


hXarr=(0.1 0.1 0.05)
Xminarr=(4.00 0.0 -5.5 0.0)
Xmaxarr=(9.00 4.0 0.0 3.0)
xi_a=(1000.)
xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
psi0arr=(0.008)
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
#SBATCH --job-name=graph$actiontime
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

python3 /home/bcheng4/TwoCapital_Shrink/abatement/${python_name} --dataname  ${action_name} --pdfname ${server_name} --psi0arr ${psi0arr[@]} --psi1arr ${psi1arr[@]}     --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job_graph.sh

