#! /bin/bash

# actiontime=1
# epsilonarraypost=(0.1)  # Computation of coarse grid and psi10.5, post
# epsilonarraypre=(0.005) # Computation of coarse grid and psi10.5, pre

actiontime=2
epsilonarraypost=(0.005) # Computation of coarse grid and psi10.8, post
epsilonarraypre=(0.005)  # Computation of coarse grid and psi10.8, pre

# actiontime=3
# epsilonarraypost=(0.008) # Computation of fine grid and psi10.5, post
# epsilonarraypre=(0.005)  # Computation of fine grid and psi10.5, pre

# actiontime=4
# epsilonarraypost=(0.005) # Computation of fine grid and psi10.8, post
# epsilonarraypre=(0.005)  # Computation of fine grid and psi10.8, pre

declare -A hXarr1=([0]=0.2 [1]=0.2 [2]=0.2)
declare -A hXarr2=([0]=0.1 [1]=0.1 [2]=0.1)
declare -A hXarr3=([0]=0.05 [1]=0.05 [2]=0.05)
# hXarrays=(hXarr1 hXarr2 hXarr3)
# hXarrays=(hXarr1)
hXarrays=(hXarr2)
# hXarrays=(hXarr3)

# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)

Xminarr=(4.00 0.0 1.0 0.0)
Xmaxarr=(9.00 4.0 6.0 3.0)

# xi_a=(0.0002 0.0002)
# xi_p=(0.1 0.075)

# xi_a=(1000. 0.0002 0.0002 0.0002 0.0002)
# xi_p=(1000. 0.1 0.075 0.05 0.025)
xi_a=(1000. 0.0002 0.0002)
xi_p=(1000. 0.050 0.025)
# xi_a=(1000.)
# xi_p=(1000.)

# psi0arr=(0.005 0.008 0.010 0.012)
# psi0arr=(0.005)
psi0arr=(0.105830)

# psi1arr=(0.5 0.6 0.7 0.8)
psi1arr=(0.5)
# psi1arr=(0.8)
psi2arr=(0.5 0.4 0.3 0.2 0.1 0.0)
# psi2arr=(0.3 0.4 0.5)
# psi2arr=(0.2)
# psi2arr=(0.2 0.1 0.0)
# psi2arr=(0.5 0.4 0.3)
# python_name_unit="Result_2jump_combine.py"
# python_name_unit="Result_2jump_combine_color_L25.py"
# python_name_unit="Result_2jump_combine_color_L_psi.py"
# python_name_unit="Result_2jump_combine_color_L_psi_rxip.py"
# python_name_unit="Result_2jump_combine_color_L_psi_rxip_fixedyaxis.py"
# python_name_unit="Result_2jump_combine_color_L_psi_rxip_smartyaxis.py"
python_name_unit="Result_2jump_combine_color_L_psi_rxip_smartyaxis_unit.py"
# python_name_unit="Result_2jump_combine_color_L_psi_2_rxip.py"
# python_name_unit="Result_2jump_combine_color_L3.py"
# python_name_unit="Result_2jump_combine_before15.py"
server_name="mercury"

LENGTH_psi=$((${#psi0arr[@]} - 1))
LENGTH_xi=$((${#xi_a[@]} - 1))

hXarr_SG=(0.2 0.2 0.2)
Xminarr_SG=(4.00 0.0 -5.5 0.0)
Xmaxarr_SG=(9.00 4.0 0.0 3.0)
interp_action_name="2jump_step_0.2_0.2_0.2_LR_0.01"
fstr_SG="NearestNDInterpolator"

auto=0

for epsilonpost in ${epsilonarraypost[@]}; do
    for hXarri in "${hXarrays[@]}"; do
        count=0
        declare -n hXarr="$hXarri"

        # action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2"
        # action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2_addmiss2"
        # action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_less2_solve0.2"
        action_name="2jump_step_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_LR_${epsilonpost}_ah_drs_unit_newpsi0"

        for PSI_0 in ${psi0arr[@]}; do
            for PSI_1 in ${psi1arr[@]}; do
                for PSI_2 in ${psi2arr[@]}; do

                    mkdir -p ./job-outs/${action_name}/PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}/

                    if [ -f ./bash/${action_name}/hX_${hXarr[0]}_PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}_Graph.sh ]; then
                        rm ./bash/${action_name}/hX_${hXarr[0]}_PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}_Graph.sh
                    fi
                    mkdir -p ./bash/${action_name}/

                    touch ./bash/${action_name}/hX_${hXarr[0]}_PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}_Graph.sh

                    tee -a ./bash/${action_name}/hX_${hXarr[0]}_PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}_Graph.sh <<EOF
#! /bin/bash


######## login 
#SBATCH --job-name=graph@
#SBATCH --output=./job-outs/${action_name}/PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}/graph_${server_name}.out
#SBATCH --error=./job-outs/${action_name}/PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}/graph_${server_name}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0


echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/${python_name_unit} --dataname  ${action_name} --pdfname ${server_name} --psi0 ${PSI_0} --psi1 ${PSI_1} --psi2 ${PSI_2} --xiaarr ${xi_a[@]} --xigarr ${xi_p[@]}   --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]} --auto $auto
# python3 /home/bcheng4/TwoCapital_Shrink/abatement/${python_name_unit} --dataname  ${action_name} --pdfname ${server_name} --psi0 ${PSI_0} --psi1 ${PSI_1} --psi2 ${PSI_2} --xiaarr ${xi_a[@]} --xigarr ${xi_p[@]}   --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

echo "Program ends \$(date)"

EOF

                    sbatch ./bash/${action_name}/hX_${hXarr[0]}_PSI0_${PSI_0}_PSI1_${PSI_1}_PSI2_${PSI_2}_Graph.sh

                done
            done
        done
    done
done
