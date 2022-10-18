#! /bin/bash

if [ -f /scratch/bincheng/bash/delaytest2.sh ]
then
		rm /scratch/bincheng/bash/delaytest2.sh
fi

touch /scratch/bincheng/bash/delaytest2.sh

tee -a /scratch/bincheng/bash/delaytest2.sh << EOF
#! /bin/bash


######## login 
#SBATCH --job-name=delay2
#SBATCH --output=/scratch/bincheng/job-outs/test/delaytest2.out
#SBATCH --error=/scratch/bincheng/job-outs/test/delaytest2.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

res1=\$(date +%s.%N)

echo "Program ends \$(date)"

# main program


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


sbatch /scratch/bincheng/bash/delaytest2.sh

echo "All Done"

