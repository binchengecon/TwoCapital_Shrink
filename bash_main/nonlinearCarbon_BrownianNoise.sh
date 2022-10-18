#! /bin/bash

#####################
# post part
#####################

NUM=100
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))

action_name="test" 

mu_T=0
mu_C=0
sigma_T=0.001
sigma_C=0.001
# please specify a new action name every time 
# otherwise the wait option won't work
# for old action name, check .abatement/data_2tech/ subfolder name are old names


count=0

for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 

		mkdir -p ./job-outs/${action_name}/${PSI_0}_${PSI_1}/

		if [ -f ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh ]
		then
				rm ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh
		fi

        mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
		
		tee -a ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=post_$count
#SBATCH --output=./job-outs/${action_name}/${PSI_0}_${PSI_1}/${server_name}_post_$i.out
#SBATCH --error=./job-outs/${action_name}/${PSI_0}_${PSI_1}/${server_name}_post_$i.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/postdamage_spe_psi_gamma_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}

echo "Program ends \$(date)"

EOF
count=$(($count+1))
		done
	done
done




for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in 0.008 0.010 0.012
	do
		for PSI_1 in 0.8
		do 
		sbatch ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_$i.sh 

		done
	done
done

#####################
# predamage part
#####################



count=$((0))


for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 

	mkdir -p ./job-outs/${action_name}/${PSI_0}_${PSI_1}/

	if [ -f ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}.sh ]
	then
			rm ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	fi

	mkdir -p ./bash/${action_name}/

	touch ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	
	tee -a ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=pre-${count}
#SBATCH --output=./job-outs/${action_name}/${PSI_0}_${PSI_1}/${server_name}_pre_$count.out
#SBATCH --error=./job-outs/${action_name}/${PSI_0}_${PSI_1}/${server_name}_pre_$count.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/predamage_spe_psi_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}

for i in $(seq 0 $ID_MAX_DAMAGE)
do
	python3 /home/bcheng4/TwoCapital_Shrink/abatement/predamage_frompost.py --xi_a 1000.0 --xi_g 1000.0 --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}
done


python3 /home/bcheng4/TwoCapital_Shrink/abatement/predamage_spe_psi_name_moreiteration.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}

echo "Program ends \$(date)"

EOF

count=$(($count+1))

	done
done



for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 
	sbatch ./bash/${action_name}/mercury_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
	done
done

#####################
# graph part
#####################



if [ -f ./bash/${action_name}/${server_name}_graph.sh ]
then
		rm ./bash/${action_name}/${server_name}_graph.sh
fi
mkdir -p ./bash/${action_name}/

touch ./bash/${action_name}/${server_name}_graph.sh

tee -a ./bash/${action_name}/${server_name}_graph.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=graph
#SBATCH --output=./job-outs/${action_name}/${server_name}_graph.out
#SBATCH --error=./job-outs/${action_name}/${server_name}_graph.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

for PSI_0 in 0.008 0.010 0.012
do
	for PSI_1 in 0.8
	do 
		python3 /home/bcheng4/TwoCapital_Shrink/abatement/graph_frompre.py --xi_a 1000.0 --xi_g 1000.0 --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name}
	done
done

python3 /home/bcheng4/TwoCapital_Shrink/abatement/Result_spe_name_moreiteration.py --dataname  $action_name --pdfname $server_name

echo "Program ends \$(date)"

EOF


sbatch ./bash/${action_name}/job_graph.sh

