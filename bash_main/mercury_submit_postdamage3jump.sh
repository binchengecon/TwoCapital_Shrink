#! /bin/bash



# action_name="repro_Suri10dmgparal_pureoriginal"
# python_name="postdamage_spe_xi_psi_gammalist_name.py"




# action_name="repro_Suri20dmgparal_pureoriginal2"
# python_name="postdamage_spe_xi_psi_gammalist_name2.py"

# # under "repro_Suri20dmgparal_pureoriginal2"
# NUM_DAMAGE=20
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.1 0.1)
# fractionarr=(0.1 0.01 0.01)
# maxiterarr=(20000 20000 20000)
# hK=0.2
# hY=0.2
# hL=0.2
# Y_max_short=3.0
# xi_a=(0.0002 0.0002 1000.)
# xi_p=(0.025 0.050 1000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# action_name="repro_Suri20dmgparal_pureoriginal2"
# python_name="postdamage_spe_xi_psi_gammalist_name2.py"
# # under "repro_Suri20dmgparal_pureoriginal2"
# NUM_DAMAGE=1
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.1 0.1)
# fractionarr=(0.2 0.01 0.01)
# maxiterarr=(5000 10000 10000)
# hK=0.2
# hY=0.2
# hL=0.2
# Y_max_short=3.0
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# action_name="rep_20dmgparal"
# python_name="postdamage_spe_xi_psi_gammalist_name2.py"
# # under "rep_20dmgparal":note Lmin=-5.5
# NUM_DAMAGE=20
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.1 0.1)
# fractionarr=(0.1 0.01 0.01)
# maxiterarr=(20000 20000 20000)
# hK=0.1
# hY=0.1
# hL=0.1
# Y_max_short=3.0
# xi_a=(10000.)
# xi_p=(10000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# action_name="rep_20dmgparal_Lmin55xia1000"
# python_name="postdamage_spe_xi_psi_gammalist_name2.py"
# # under "rep_20dmgparal_Lmin55xia1000":note Lmin=-5.5
# NUM_DAMAGE=20
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.1 0.1)
# fractionarr=(0.1 0.01 0.01)
# maxiterarr=(20000 20000 20000)
# hK=0.1
# hY=0.1
# hL=0.1
# Y_max_short=3.0
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# action_name="rep_10dmgparal_Lmin5xia1000"
# python_name="postdamage_spe_xi_psi_gammalist_name3.py"
# # under "rep_10dmgparal_Lmin5xia1000":note Lmin=-5.0
# NUM_DAMAGE=10
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.1 0.1)
# fractionarr=(0.1 0.01 0.01)
# maxiterarr=(30000 30000 30000)
# hK=0.1
# hY=0.1
# hL=0.1
# Y_max_short=3.0
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.005)
# psi1arr=(0.5)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



action_name="rep_10dmgparal_Lmin5xia10000"
python_name="postdamage_spe_xi_psi_gammalist_name3.py"
# under "rep_10dmgparal_Lmin5xia1000":note Lmin=-5.0
# job name 1e4
NUM_DAMAGE=10
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
epsilonarr=(0.1 0.1 0.1)
fractionarr=(0.1 0.01 0.01)
maxiterarr=(30000 30000 30000)
hK=0.1
hY=0.1
hL=0.1
Y_max_short=3.0
xi_a=(10000.)
xi_p=(10000.)
psi0arr=(0.005)
psi1arr=(0.5)
LENGTH_xi=$((${#xi_a[@]}-1))
count=0


# Actually psi0 should be 0.005 instead of 0.010 
# This guess can also be verified by checkSuriConvergtence.py, 
# where psi0=0.010 gievs error at 0.002 and psi0=0.005 gives error at 1e-7or8.


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
#SBATCH --job-name=1e4post_$count
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

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hK $hK --hY  	$hY --hL  $hL

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
