#! /bin/bash

actiontime=1
epsilonarraypost=(0.1) # Computation of fine grid and psi10.8, post
# epsilonarraypost=(0.05) # Computation of fine grid and psi10.8, post

NUM_DAMAGE=3

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
# hXarrays=(hXarr1 hXarr2 hXarr3)
hXarrays=(hXarr1)
# hXarrays=(hXarr2)
# hXarrays=(hXarr3)


Xminarr=(4.00 0.0 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)


xi_a=(0.0004 0.0002 0.0001 0.00005 0.0004 0.0002 0.0001 0.00005 1000.)
xi_p=(0.025 0.025 0.025 0.025 0.050 0.050 0.050 0.050 1000.)

# xi_a=(0.0004 0.0002 0.0001 0.00005)
# xi_p=(0.025 0.025 0.025 0.025)


psi0arr=(0.105830)
# psi0arr=(0.000001)
psi1arr=(0.5)


# psi2arr=(0.0 0.1 0.2 0.3 0.4 0.5)
# psi2arr=(0.5)


python_name_unit="Result_2jump_UD_simulate_CRS.py"
# python_name_unit="Result_2jump_UD_simulate_CRS_MulJump.py"


server_name="mercury"

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

auto=1
year=25
# year=40

# scheme_array=("macroannual" "newway" "newway" "newway" "check")
# HJBsolution_array=("simple" "iterative_partial" "iterative_fix" "n_iterative_fix" "iterative_partial")

# scheme_array=("macroannual" "newway" "newway" "newway")
# HJBsolution_array=("simple" "iterative_partial" "iterative_fix" "n_iterative_fix")
# scheme_array=("newway")
# HJBsolution_array=("n_iterative_fix")

# scheme_array=("newway" "newway" "newway" "check")
# HJBsolution_array=("iterative_partial" "iterative_fix" "n_iterative_fix" "iterative_partial")
# scheme_array=("newway" "check")
# HJBsolution_array=("iterative_fix" "iterative_partial")

scheme_array=("direct")
HJBsolution_array=("direct")


LENGTH_scheme=$((${#scheme_array[@]} - 1))




for epsilonpost in ${epsilonarraypost[@]}; do
    for hXarri in "${hXarrays[@]}"; do
        count=0
        declare -n hXarr="$hXarri"

        action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_CRS"
        # action_name="2jump_step_${Xminarr[0]},${Xmaxarr[0]}_${Xminarr[1]},${Xmaxarr[1]}_${Xminarr[2]},${Xmaxarr[2]}_SS_${hXarr[0]},${hXarr[1]},${hXarr[2]}_LR_${epsilonpost}_CRS_MulJump"

        for PSI_0 in ${psi0arr[@]}; do
            for PSI_1 in ${psi1arr[@]}; do
                    for j in $(seq 0 $LENGTH_xi); do
                        for k in $(seq 0 $LENGTH_scheme); do

                    mkdir -p ./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/

                    if [ -f ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh ]; then
                        rm ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh
                    fi
                    mkdir -p ./bash/${action_name}/

                    touch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh

                    tee -a ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=sim_${year}
#SBATCH --output=./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/graph_simulate_${year}.out
#SBATCH --error=./job-outs/${action_name}/Graph_Simulate/scheme_${scheme_array[$k]}_HJB_${HJBsolution_array[$k]}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/graph_simulate_${year}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 /home/bcheng4/TwoCapital_Shrink/abatement_UD/${python_name_unit} --dataname  ${action_name} --pdfname ${server_name} --psi0 ${PSI_0} --psi1 ${PSI_1} --xiaarr ${xi_a[$j]} --xigarr ${xi_p[$j]}   --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]} --auto $auto --IntPeriod ${year} --scheme ${scheme_array[$k]}  --HJB_solution ${HJBsolution_array[$k]}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF

                    sbatch ./bash/${action_name}/hX_${hXarr[0]}_xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}_Graph.sh

                        done
                    done
                done
            done
    done
done