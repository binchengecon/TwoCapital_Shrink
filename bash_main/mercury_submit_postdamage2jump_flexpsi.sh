#! /bin/bash

# epsilon=0.007

# actiontime=1
# action_name="2jump_step_0.05_0.05_0.05_LR_${epsilon}_reportmore"
# python_name="postdamage_2jump_timereport.py"
# NUM_DAMAGE=10
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 ${epsilon})
# fractionarr=(0.1 ${epsilon})
# maxiterarr=(60000 100000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)

# hXarr_SG=(0.2 0.2 0.2)
# Xminarr_SG=(4.00 0.0 -5.5 0.0)
# Xmaxarr_SG=(9.00 4.0 0.0 3.0)
# interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
# fstr_SG="NearestNDInterpolator"

# xi_a=(1000.)
# xi_p=(1000.)

# psi0arr=(0.008 )
# # psi0arr=(0.005)
# psi1arr=(0.8)
# # psi1arr=(0.5)
# LENGTH_psi=$((${#psi0arr[@]}-1))
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

epsilon=0.01
actiontime=2
action_name="2jump_step_0.2_0.2_0.2_LR_${epsilon}_reportmore"
python_name="postdamage_2jump_timereport.py"
NUM_DAMAGE=10
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
epsilonarr=(0.1 ${epsilon})
fractionarr=(0.1 ${epsilon})
maxiterarr=(60000 100000)
hXarr=(0.2 0.2 0.2)
Xminarr=(4.00 0.0 -5.5 0.0)
Xmaxarr=(9.00 4.0 0.0 3.0)

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

xi_a=(1000.)
xi_p=(1000.)

psi0arr=(0.008 )
# psi0arr=(0.005)
psi1arr=(0.8)
# psi1arr=(0.5)
LENGTH_psi=$((${#psi0arr[@]}-1))
LENGTH_xi=$((${#xi_a[@]}-1))
count=0





for k in $(seq 0 $LENGTH_psi) 
do
	for i in $(seq 0 $ID_MAX_DAMAGE)
	do
		for j in $(seq 0 $LENGTH_xi)
		do

		mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}/

		if [ -f ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}_ID_${i}.sh ]
		then
				rm ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}_ID_${i}.sh
		fi

        mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}_ID_${i}.sh
		
		tee -a ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}_ID_${i}.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=pt${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}/mercury_post_$i.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}/mercury_post_$i.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 ${psi0arr[$k]} --psi_1 ${psi1arr[$k]} --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]} --hXarr_SG ${hXarr_SG[@]} --Xminarr_SG ${Xminarr_SG[@]} --Xmaxarr_SG ${Xmaxarr_SG[@]} --fstr_SG ${fstr_SG} --interp_action_name ${interp_action_name}

echo "Program ends \$(date)"

EOF
count=$(($count+1))
		done
	done
done


count=0

for k in $(seq 0 $LENGTH_psi) 
do
	for j in $(seq 0 $LENGTH_xi)
	do
		for i in $(seq 0 $ID_MAX_DAMAGE)
		do
		
		sbatch ./bash/${action_name}/mercury_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${psi0arr[$k]}_PSI1_${psi1arr[$k]}_ID_${i}.sh 
		count=$(($count+1))

		done
	done
done










