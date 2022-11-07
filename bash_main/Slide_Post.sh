#! /bin/bash

# Coarse Grid: PSI1 = 0.5 0.8 done
# Find Grid: TBD

# epsilonarray=(0.005 0.008 0.012 0.1)
# epsilonarray=(0.1) #Computation of coarse grid and psi10.5
epsilonarray=(0.008) #Computation of coarse grid and psi10.8

# epsilonarray=(0.1 0.008) # Computation of fine grid and psi10.5 test 0.1 and 0.008 work or not
# epsilonarray=(0.005 0.008) # Computation of fine grid and psi10.8 test 0.005 and 0.008 work or not
# epsilonarray=(0.005) # Computation of fine grid and psi10.8 test 0.005 and 0.008 work or not

actiontime=1
python_name="postdamage_2jump_repless.py"

NUM_DAMAGE=10

ID_MAX_DAMAGE=$((NUM_DAMAGE - 1))

maxiterarr=(60000 100000)

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
# hXarrays=(hXarr1 hXarr2 hXarr3)
# hXarrays=(hXarr1)
hXarrays=(hXarr3)

Xminarr=(4.00 0.0 -5.5 0.0)
Xmaxarr=(9.00 4.0 0.0 3.0)

# xi_a=(1000. 0.0002 0.0002)
# xi_p=(1000. 0.05 0.025)
xi_a=(1000.)
xi_p=(1000.)

psi0arr=(0.005 0.008 0.010 0.012)
# psi1arr=(0.5 0.6 0.7 0.8)
# psi1arr=(0.5)
psi1arr=(0.8)

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

for epsilon in ${epsilonarray[@]}; do
	for hXarri in "${hXarrays[@]}"; do
		count=0
		declare -n hXarr="$hXarri"

		action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilon}_Psi01ComparisonSlide"

		epsilonarr=(0.1 ${epsilon})
		fractionarr=(0.1 ${epsilon})

		for i in $(seq 0 $ID_MAX_DAMAGE); do
			for PSI_0 in ${psi0arr[@]}; do
				for PSI_1 in ${psi1arr[@]}; do
					for j in $(seq 0 $LENGTH_xi); do

						mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

						if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh ]; then
							rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
						fi

						mkdir -p ./bash/${action_name}/

						touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh

						tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${hXarr[0]}_$i
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
						count=$(($count + 1))
						sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
					done
				done
			done
		done
	done
done

# # Coarse Grid: PSI1 = 0.5 0.8 done
# # Find Grid: TBD

# # epsilonarray=(0.005 0.008 0.012 0.1)
# # epsilonarray=(0.1) #Computation of coarse grid and psi10.5
# # epsilonarray=(0.008) #Computation of coarse grid and psi10.8

# # epsilonarray=(0.1 0.008) # Computation of fine grid and psi10.5 test 0.1 and 0.008 work or not
# # epsilonarray=(0.008) # Computation of fine grid and psi10.5 test 0.005 work or not
# # epsilonarray=(0.005) # Computation of fine grid and psi10.8 test 0.008 work or not
# epsilonarray=(0.008) # Computation of fine grid and psi10.8 test 0.008 work or not

# actiontime=1
# python_name="postdamage_2jump_repless_interp_SG.py"

# NUM_DAMAGE=10

# ID_MAX_DAMAGE=$((NUM_DAMAGE - 1))

# maxiterarr=(60000 100000)
# # maxiterarr=(10 10)

# declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
# declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
# declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
# # hXarrays=(hXarr1 hXarr2 hXarr3)
# # hXarrays=(hXarr1)
# hXarrays=(hXarr3)

# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)

# # xi_a=(1000. 0.0002 0.0002)
# # xi_p=(1000. 0.05 0.025)
# xi_a=(1000.)
# xi_p=(1000.)

# psi0arr=(0.005 0.008 0.010 0.012)
# # psi1arr=(0.5 0.6 0.7 0.8)
# # psi1arr=(0.5)
# psi1arr=(0.8)

# LENGTH_psi=$((${#psi0arr[@]} - 1))
# LENGTH_xi=$((${#xi_a[@]} - 1))

# hXarr_SG=(0.2 0.2 0.2)
# Xminarr_SG=(4.00 0.0 -5.5 0.0)
# Xmaxarr_SG=(9.00 4.0 0.0 3.0)
# # interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.1_Psi01ComparisonSlide"
# interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.008_Psi01ComparisonSlide"
# fstr_SG="NearestNDInterpolator"

# for epsilon in ${epsilonarray[@]}; do
# 	for hXarri in "${hXarrays[@]}"; do
# 		count=0
# 		declare -n hXarr="$hXarri"

# 		action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilon}_Psi01ComparisonSlide_Interpolate"

# 		epsilonarr=(0.1 ${epsilon})
# 		fractionarr=(0.1 ${epsilon})

# 		for i in $(seq 0 $ID_MAX_DAMAGE); do
# 			for PSI_0 in ${psi0arr[@]}; do
# 				for PSI_1 in ${psi1arr[@]}; do
# 					for j in $(seq 0 $LENGTH_xi); do

# 						mkdir -p ./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

# 						if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh ]; then
# 							rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
# 						fi

# 						mkdir -p ./bash/${action_name}/

# 						touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh

# 						tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh <<EOF
# #! /bin/bash

# ######## login
# #SBATCH --job-name=${hXarr[0]}_$i
# #SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.out
# #SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.err

# #SBATCH --account=pi-lhansen
# #SBATCH --partition=standard
# #SBATCH --cpus-per-task=3
# #SBATCH --mem=16G
# #SBATCH --time=7-00:00:00

# ####### load modules
# module load python/booth/3.8/3.8.5  gcc/9.2.0

# echo "\$SLURM_JOB_NAME"

# echo "Program starts \$(date)"
# start_time=\$(date +%s)
# # perform a task

# python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}  --hXarr_SG ${hXarr_SG[@]} --Xminarr_SG ${Xminarr_SG[@]} --Xmaxarr_SG ${Xmaxarr_SG[@]} --fstr_SG ${fstr_SG} --interp_action_name ${interp_action_name}

# echo "Program ends \$(date)"
# end_time=\$(date +%s)

# # elapsed time with second resolution
# elapsed=\$((end_time - start_time))

# eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

# EOF
# 						count=$(($count + 1))
# 						sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_ID_${i}.sh
# 					done
# 				done
# 			done
# 		done
# 	done
# done
