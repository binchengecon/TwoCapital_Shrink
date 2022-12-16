#! /bin/bash

action_name="UploadDB_Parallel"
python_name="UploadDB4.py"
MotherRoot="/scratch/bincheng/abatement/data_2tech/"
DaughterRoot="/climatemodeling/IMSI_Mitigation/data/abatement_v2/"

access_token="sl.BSu-dl4W2cFpHnyZddY3CGaT_WSbGChHCPV_AngpclevKi1Jrngs92ZW68ZOqDOei2cylDnsCN7d57zCrCYvrD7kazWMZdEwGuBR6Jlwi-snLA1mP97egfS1-Aot7DnPzbG9LtocVcWi"
refresh_token="et47BKvSlxMAAAAAAAAAAeajinfYkuBJLBZ4f9_aJcCAJs9h_SYSvcyxPpLvsfZ8"

declare -a FolderArray=("2jump_step_0.05_0.05_0.05_LR_0.005_Psi01ComparisonSlide"
    "2jump_step_0.05_0.05_0.05_LR_0.005_Psi01ComparisonSlide_Interpolate"
    "2jump_step_0.05_0.05_0.05_LR_0.007_reportmore"
    "2jump_step_0.05_0.05_0.05_LR_0.008_Psi01ComparisonSlide"
    "2jump_step_0.05_0.05_0.05_LR_0.008_Psi01ComparisonSlide_Interpolate"
    "2jump_step_0.05_0.05_0.05_LR_0.1_Psi01ComparisonSlide"
    "2jump_step_0.2_0.2_0.2_LR_0.005_Psi01ComparisonSlide"
    "2jump_step_0.2_0.2_0.2_LR_0.008_Psi01ComparisonSlide"
    "2jump_step_0.2_0.2_0.2_LR_0.008_Psi01ComparisonSlide_test"
    "2jump_step_0.2_0.2_0.2_LR_0.1_Psi01ComparisonSlide")

# Read the array values with space

count=0
for Folder in "${FolderArray[@]}"; do
    mkdir -p ./job-outs/${action_name}/

    if [ -f ./bash/${action_name}/${Folder}.sh ]; then
        rm ./bash/${action_name}/${Folder}.sh
    fi

    mkdir -p ./bash/${action_name}/

    touch ./bash/${action_name}/${Folder}.sh

    tee -a ./bash/${action_name}/${Folder}.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=UDB${count}
#SBATCH --output=./job-outs/${action_name}/${Folder}.out
#SBATCH --error=./job-outs/${action_name}/${Folder}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task


python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --Folder ${Folder} --MotherRoot ${MotherRoot} --DaughterRoot ${DaughterRoot}  --access_token ${access_token} --refresh_token ${refresh_token}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
    count=$(($count + 1))
    sbatch ./bash/${action_name}/${Folder}.sh
done
