#! /bin/bash


NUM_DAMAGE=6
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))

action_name="moreiteration"

count=0

for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 

		mkdir -p ./job-outs/${PSI_0}_${PSI_1}/

		if [ -f ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh ]
		then
				rm ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		fi

      mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
		
		tee -a ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=test_$count
#SBATCH --output=./job-outs/${action_name}/${PSI_0}_${PSI_1}/test_$i.out
#SBATCH --error=./job-outs/${action_name}/${PSI_0}_${PSI_1}/test_$i.err

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

python /home/bincheng/TwoCapital_Bin/abatement/postdamage_spe_psi_gamma_name.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}

echo "Program ends \$(date)"

EOF
count=$(($count+1))
		done
	done
done




# for i in $(seq 0 $ID_MAX_DAMAGE)
# do
# 	for PSI_0 in 0.008 0.010 0.012
# 	do
# 		for PSI_1 in 0.8
# 		do 
# 		sbatch ./bash/${action_name}/job2_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh 

# 		done
# 	done
# done
