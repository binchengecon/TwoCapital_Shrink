#! /bin/bash

actiontime=1
# epsilonarraypost=(0.005) # Computation of fine grid and psi10.8, post
epsilonarraypost=(0.1) # Computation of fine grid and psi10.8, post
# epsilonarraypre=(0.1 0.05 0.03 0.02 0.01) # Computation of fine grid and psi10.5, pre
# epsilonarraypre=(0.0075 0.0065 0.005 0.004 0.003 0.002 0.001) # Computation of fine grid and psi10.8, pre
epsilonarraypre=(0.005) #
# epsilonarraypre=(0.1) #

python_name="predamage_2jump_drs_unit_ambplus.py"
# python_name="predamage_2jump_ah_drs_addmiss2.py"

NUM_DAMAGE=5

ID_MAX_DAMAGE=$((NUM_DAMAGE - 1))

maxiterarr=(80000 200000)

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
# hXarrays=(hXarr1 hXarr2 hXarr3)
hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)

# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)

Xminarr=(4.00 0.0 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)

# xi_a=(0.0002 0.0002)
# xi_p=(0.1 0.075)
# xi_a=(1000. 0.0002 0.0002 0.0002 0.0002)
# xi_p=(1000. 0.1 0.075 0.05 0.025)
# xi_a=(0.0002 0.0002 0.0002 0.0002)
# xi_p=(0.1 0.075 0.05 0.025)
# xi_a=(1000. 0.0002 0.0002)
# xi_p=(1000. 0.05 0.025)

# xi_a=(1000. 0.0002 0.0002)
# xi_p=(1000. 0.050 0.025)
# xi_a=(0.0002 0.0004 0.0006 0.0008 0.0010 0.0002 0.0004 0.0006 0.0008 0.0010)
# xi_p=(0.050 0.050 0.050 0.050 0.050 0.025 0.025 0.025 0.025 0.025)

# xi_a=(0.00002 0.00004 0.00006 0.00008 0.00010 0.00002 0.00004 0.00006 0.00008 0.00010)
# xi_p=(0.050 0.050 0.050 0.050 0.050 0.025 0.025 0.025 0.025 0.025)

xi_a=(0.0012 0.00125 0.0013 0.0014 0.0015)
xi_p=(0.050 0.050 0.050 0.050 0.050)

# psi0arr=(0.005 0.008 0.010 0.012)
# psi0arr=(0.005)
psi0arr=(0.105830)
# psi0arr=(1.2 0.5 0.005)

# psi1arr=(0.5 0.6 0.7 0.8)
psi1arr=(0.5)
# psi1arr=(0.8)
# psi2arr=(0.5 0.4 0.3 0.2)
# psi2arr=(0.5 0.4)
# psi2arr=(0.2)
# psi2arr=(0.5 0.4 0.3 0.2 0.1 0.0)
# psi2arr=(0.5 0.4 0.3)
# psi2arr=(0.2 0.1 0.0)

# sarr=(0.01 0.02 0.03)
# tauarr=(0.01 0.02 0.03)
# Trarr=(0.01 0.02 0.03)

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))
# LENGTH_s=$((${#sarr[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

for epsilon in ${epsilonarraypre[@]}; do
	for epsilonpost in ${epsilonarraypost[@]}; do
		for hXarri in "${hXarrays[@]}"; do
			count=0
			declare -n hXarr="$hXarri"

			# action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs"
			# action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2"
			# action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2_addmiss2"
			# action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2_solve0.2"
			action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_drs_unit_ambplus_calibxia"

			epsilonarr=(0.05 ${epsilon})
			fractionarr=(0.1 ${epsilon})
			for PSI_0 in ${psi0arr[@]}; do
				for PSI_1 in ${psi1arr[@]}; do
					for j in $(seq 0 $LENGTH_xi); do

						mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

						if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Eps_${epsilon}.sh ]; then
							rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Eps_${epsilon}.sh
						fi

						mkdir -p ./bash/${action_name}/

						touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Eps_${epsilon}.sh

						tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Eps_${epsilon}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${xi_p[$j]}_${epsilon}
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_pre_${epsilon}.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_pre_${epsilon}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_p ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --psi_0 $PSI_0 --psi_1 $PSI_1    --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
						count=$(($count + 1))
						sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Eps_${epsilon}.sh
					done
				done
			done
		done
	done
done
