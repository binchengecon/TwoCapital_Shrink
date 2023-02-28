#! /bin/bash

epsilonarray=(0.1) # 0.2
# epsilonarray=(0.05) # 0.1
# epsilonarray=(0.025) # 0.1
# epsilonarray=(0.01) # 0.1
# epsilonarray=(0.005) # 0.05

python_name="postdamage_2jump_Result_support.py"

# NUM_DAMAGE=4
NUM_DAMAGE=4

ID_MAX_DAMAGE=$((NUM_DAMAGE - 1))

maxiterarr=(80000 200000)

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
declare -A hXarr4=([0]=0.2 [1]=0.05 [2]=0.2)
# hXarrays=(hXarr1 hXarr2 hXarr3)
hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)
# hXarrays=(hXarr4)

# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)

Xminarr=(4.00 0.0 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)


# enlarging logJg: doesn't seem to work
# Xminarr=(4.00 0.0 0.0 0.0)
# Xmaxarr=(9.00 4.0 7.0 3.0)


# enlarging logK: not converging
# Xminarr=(3.00 0.0 1.0 0.0)
# Xmaxarr=(10.00 4.0 6.0 3.0)

# enlarging logK: 
# Xminarr=(4.00 0.0 1.0 0.0)
# Xmaxarr=(10.00 4.0 6.0 3.0)



xi_a=(0.0004 0.0002 0.0001 0.00005)
xi_p=(0.025 0.025 0.025 0.025)

xi_a_support=(100000.)
xi_g_support=(100000.)
theta_ell_type=("full")

psi0arr=(0.105830)
psi1arr=(0.5)

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))
LENGTH_xi_support=$((${#xi_a_support[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

for epsilon in ${epsilonarray[@]}; do
	for hXarri in "${hXarrays[@]}"; do
		count=0
		declare -n hXarr="$hXarri"

		action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilon}"

		epsilonarr=(0.1 ${epsilon})
		fractionarr=(0.1 ${epsilon})

		for i in $(seq 0 $ID_MAX_DAMAGE); do
			for PSI_0 in ${psi0arr[@]}; do
				for PSI_1 in ${psi1arr[@]}; do
					for j in $(seq 0 $LENGTH_xi); do
					for jj in $(seq 0 $LENGTH_xi_support); do

						mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

						if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Support_${jj}_ID_${i}.sh ]; then
							rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Support_${jj}_ID_${i}.sh
						fi

						mkdir -p ./bash/${action_name}/

						touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Support_${jj}_ID_${i}.sh

						tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Support_${jj}_ID_${i}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${xi_p[$j]}_$i
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/support_${jj}_gamma_$i.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/support_${jj}_gamma_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=mcn53

####### load modules
module load python/booth/3.8  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --xi_a_support ${xi_a_support[$jj]} --xi_g_support ${xi_g_support[$jj]}  --theta_ell_type ${theta_ell_type[$jj]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"
echo ${hXarr[@]}

EOF
						count=$(($count + 1))
						sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Support_${jj}_ID_${i}.sh
					done
				done
			done
		done
	done
done
done