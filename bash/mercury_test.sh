#! /bin/bash

if [ -f ./bash/delaytest.sh ]
then
		rm ./bash/delaytest.sh
fi


touch ./bash/delaytest.sh

tee -a ./bash/delaytest.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=delay1
#SBATCH --output=./job-outs/old/delaytest.out
#SBATCH --error=./job-outs/old/delaytest.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

res1=\$(date +%s.%N)

echo "Program starts \$(date)"

# main program
sleep 10s

python3 ./abatement/mercury_test.py

echo "Program ends \$(date)"

res2=\$(date +%s.%N)
dt=\$(echo "\$res2 - \$res1" | bc)
dd=\$(echo "\$dt/86400" | bc)
dt2=\$(echo "\$dt-86400*\$dd" | bc)
dh=\$(echo "\$dt2/3600" | bc)
dt3=\$(echo "\$dt2-3600*\$dh" | bc)
dm=\$(echo "\$dt3/60" | bc)
ds=\$(echo "\$dt3-60*\$dm" | bc)

echo "Program ends time \${dd} day \${dh} hour \${dm} minute \${ds} second"

EOF


sbatch ./bash/delaytest.sh

echo "All Done"

