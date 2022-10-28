# actiontime=1
# epsilonarraypost=(0.1)  # Computation of coarse grid and psi10.5, post
# epsilonarraypre=(0.005) # Computation of coarse grid and psi10.5, pre

# actiontime=2
# epsilonarraypost=(0.008) # Computation of coarse grid and psi10.8, post
# epsilonarraypre=(0.005)  # Computation of coarse grid and psi10.8, pre

# actiontime=3
epsilonarraypost=(0.008) # Computation of fine grid and psi10.5, post
# epsilonarraypre=(0.005)  # Computation of fine grid and psi10.5, pre

actiontime=4
# epsilonarraypost=(0.005) # Computation of fine grid and psi10.8, post
epsilonarraypre=(0.005) # Computation of fine grid and psi10.8, pre

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
psi1arr=(0.5)
# psi1arr=(0.8)
python_name_unit="Result_2jump_unit.py"
server_name="mercury"

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

for epsilonpost in ${epsilonarraypost[@]}; do
    for hXarri in "${hXarrays[@]}"; do
        count=0
        declare -n hXarr="$hXarri"

        action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_Psi01ComparisonSlide"

        for PSI_0 in ${psi0arr[@]}; do
            for PSI_1 in ${psi1arr[@]}; do
                for j in $(seq 0 $LENGTH_xi); do

                    if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh ]; then
                        rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh
                    fi
                    mkdir -p ./bash/${action_name}/

                    touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh

                    tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=graph$actiontime
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/graph_${server_name}.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/graph_${server_name}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/${python_name_unit} --dataname  ${action_name} --pdfname ${server_name} --psi0 ${PSI_0} --psi1 ${PSI_1}     --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"

EOF

                    sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh

                done
            done
        done
    done
done
