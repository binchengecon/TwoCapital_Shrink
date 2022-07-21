#! /bin/bash




action_name="repro_Suri10dmgparal_pureoriginal"
python_name="predamage_spe_xi_psi_gammalist_name.py"



NUM_DAMAGE=10
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
xi_a=(0.0002 0.0002 1000.)
xi_p=(0.025 0.050 1000.)
# xi_a=(1000.)
# xi_p=(1000.)
psi0arr=(0.005)
psi1arr=(0.5)
LENGTH_xi=$((${#xi_a[@]}-1))
count=0
hK=0.2
hY=0.2
hL=0.2
Y_max_short=3.0



for PSI_0 in ${psi0arr[@]}
do
	for PSI_1 in ${psi1arr[@]}
	do 
	for j in $(seq 0 $LENGTH_xi)
	do

		mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

		if [ -f ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}.sh ]
		then
				rm ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
		fi

        mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}.sh
		
		tee -a ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=10pre_$count
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_pre.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_pre.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]} --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hK $hK --hY $hY --hL $hL

echo "Program ends \$(date)"

EOF
count=$(($count+1))
	done
	done
done






for PSI_0 in ${psi0arr[@]}
do
	for PSI_1 in ${psi1arr[@]}
	do 
	for j in $(seq 0 $LENGTH_xi)
	do
	sbatch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}.sh 

	done
	done
done
