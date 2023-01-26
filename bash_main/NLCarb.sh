#! /bin/bash

epsilonarray=(0.1) #Computation of coarse grid and psi10.5
actiontime=1

python_name="HJB2.py"

maxiter=80000

declare -A hXarr1=([0]=0.1 [1]=10.0 [2]=50.0)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)

hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)

ceartharray=(0.1 3.2 10 40 80 160 640 2560)
taucarray=(0.1 1 10 50 80)

Xminarr=(0.00 10.0 10.0)
Xmaxarr=(3.00 100.0 1000.0)

deltaarr=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for epsilon in ${epsilonarray[@]}; do
	for hXarri in "${hXarrays[@]}"; do
		count=0
		declare -n hXarr="$hXarri"

		action_name="NonlinearCarbon_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilon}"


		epsilonarr=${epsilon}
		fraction=${epsilon}

		for delta in ${deltaarr[@]}; do
		for cearth in ${ceartharray[@]}; do
		for tauc in ${taucarray[@]}; do

		mkdir -p ./job-outs/${action_name}/

		if [ -f ./bash/${action_name}/hX_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_delta_${delta}_cearth_${cearth}_tauc_${tauc}.sh ]; then
			rm ./bash/${action_name}/hX_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_delta_${delta}_cearth_${cearth}_tauc_${tauc}.sh
		fi

		mkdir -p ./bash/${action_name}/

		touch ./bash/${action_name}/hX_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_delta_${delta}_cearth_${cearth}_tauc_${tauc}.sh

		tee -a ./bash/${action_name}/hX_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_delta_${delta}_cearth_${cearth}_tauc_${tauc}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=test
#SBATCH --output=./job-outs/${action_name}/${delta}_${cearth}_${tauc}.out
#SBATCH --error=./job-outs/${action_name}/${delta}_${cearth}_${tauc}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /home/bcheng4/TwoCapital_Shrink/nonlinearCarbon/$python_name  --epsilon ${epsilon[@]}  --fraction ${fraction[@]}   --maxiter ${maxiter[@]}  --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]} --delta ${delta} --cearth ${cearth} --tauc ${tauc}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
		count=$(($count + 1))
		sbatch ./bash/${action_name}/hX_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_delta_${delta}_cearth_${cearth}_tauc_${tauc}.sh
		done
	done
done
done
done