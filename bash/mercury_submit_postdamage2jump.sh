#! /bin/bash



# actiontime=1
# action_name="2jump_beforemayversion"
# python_name="postdamage_2jump.py"

# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.005)
# maxiterarr=(50000 50000)

# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


actiontime=2
action_name="2jump_cobwebversion"
python_name="postdamage_2jump_cobweb.py"

NUM_DAMAGE=3
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
epsilonarr=(0.1 0.01)
fractionarr=(0.1 0.01)
maxiterarr=(50000 50000)

hXarr=(0.2 0.2 0.2)
Xminarr=(4.00 0.0 -5.5 0.0)
Xmaxarr=(9.00 4.0 0.0 3.0)
xi_a=(1000.)
xi_p=(1000.)
psi0arr=(0.008 0.010 0.012)
psi1arr=(0.8)
LENGTH_xi=$((${#xi_a[@]}-1))
count=0




for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in ${psi0arr[@]}
	do
		for PSI_1 in ${psi1arr[@]}
		do 
		for j in $(seq 0 $LENGTH_xi)
		do

		mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

		if [ -f ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh ]
		then
				rm ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
		fi

        mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
		
		tee -a ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=post${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"

EOF
count=$(($count+1))
		done
		done
	done
done




for i in $(seq 0 $ID_MAX_DAMAGE)
do
	for PSI_0 in ${psi0arr[@]}
	do
		for PSI_1 in ${psi1arr[@]}
		do 
		for j in $(seq 0 $LENGTH_xi)
		do
		sbatch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh 

		done
		done
	done
done
