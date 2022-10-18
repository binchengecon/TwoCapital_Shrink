#! /bin/bash


action_name="moreiteration"

count=0



for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 

	mkdir -p ./job-outs/${action_name}/${PSI_0}_${PSI_1}/


	if [ -f ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh ]
	then
			rm ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	fi

	mkdir -p ./bash/${action_name}/

	touch ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	
	tee -a ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=pre-${count}
#SBATCH --output=./job-outs/${action_name}/${PSI_0}_${PSI_1}/pre.out
#SBATCH --error=./job-outs/${action_name}/${PSI_0}_${PSI_1}/pre.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=36:00:00

####### load modules
module load python/anaconda-2020.02
module load gcc/6.1

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

python /home/bincheng/TwoCapital_Bin/abatement/predamage_spe_psi_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}

echo "Program ends \$(date)"

EOF
count=$(($count+1))
	done
done



for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 
	sbatch ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	done
done
