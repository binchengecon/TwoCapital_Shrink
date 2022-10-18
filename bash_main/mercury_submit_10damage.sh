#! /bin/bash



action_name="repro_Suri10dmg"

xi_p=(0.025 0.050 1000.0)
xi_a=(0.0002 0.0002 1000.0)

for i in 0 1 2
do 

mkdir -p ./job-outs/${action_name}/

if [ -f ./bash/${action_name}/mercury_$i.sh ]
then
		rm ./bash/${action_name}/mercury_$i.sh
fi

mkdir -p ./bash/${action_name}/

touch ./bash/${action_name}/mercury_$i.sh

tee -a ./bash/${action_name}/mercury_$i.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=repro_$i
#SBATCH --output=./job-outs/${action_name}/mercury_$i.out
#SBATCH --error=./job-outs/${action_name}/mercury_$i.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/abatement.py --xi_p ${xi_p[$i]} --xi_a ${xi_a[$i]}

echo "Program ends \$(date)"

EOF

done




for i in 0 1 2 
do
		sbatch ./bash/${action_name}/mercury_$i.sh  
done
