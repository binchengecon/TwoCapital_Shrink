#! /bin/bash


# comment about action 3:
# first, the result is exactly the same as previous post_moreiteration.py 
# (compare job-outs/2jump_step02verify_new_comparewithold and */2jump_step02verify_old)
# so there  is no need to worry about creating all these post2jump_2jump.py
# configuration is passed NICE!


# comment about action 4-6:
# These 3 actions is made after verification of the result in action 3.
# I decrease the step size to 0.1 and 0.05.
# However, things didn't go well with 0.05,
# it turns out that with 0.05, the solution would be converging before x gradually drop to 1e+17
# but weirdly, x would suddenly explode to 1e+47, killing the computation completely.
# so first solution is to set an upper bound, of which the result is not good for the moment
# second solution is to change epsilon and fraction to stabilize it.

# comment about action7-9: 
# part 1: These 3 actions is to try out the improvement I made on original solver
# which is to impose a hard upper bound on x, R&D investment as 0.1
# due to the fact that optimal x is around 0.01
# so it's like skip directly thousands of iteration and directly somewhere near to 
# optimal solution, also avoid the explosion that we can observe in original solver with step 0.005
# Then the difference of 3 actions is step size
# I think if we had a step size=0.05, choppy pattern would be resolved and problem in action 4-6 avioded.

# (NTS) part 2: the problem is that in original solver, sometimes, x.min>0.1, but I also mute it. That is tricky.
# cause the cobweb iteration scheme determine that x = conv(x_hardbound,x_original(which is 1e+17))


# comment about 10-12:
# here we employ both the upper bound and cobweb.
# actually the reason we use upper bound is that cobweb
# need some bounding to make it work
# otherwise it is kept at x=1e+17 and produce delta/consumption extremely weird. 
# The only way that worked to escape that is by assigning upper bound at 0.1, even 1 would fail.

# comment about 14: 
# based on failure in action 4-6, with stepsize=0.05
# the computation start to explode, I tried smaller epsilon=0.005=fraction to stabilize it.
# hope this one would work

# comment about 15:
# maybe try to use original algorithm(2jump_new) as intial guess
# this way we can avoid smartx, a not well functioning head cutting




# actiontime=1
# action_name="2jump_beforemayversion"
# python_name="postdamage_2jump.py"

# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.005)
# maxiterarr=(50000 50000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# actiontime=2
# action_name="2jump_cobwebversion"
# python_name="postdamage_2jump_cobweb.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(50000 50000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0




# actiontime=3
# action_name="2jump_step02verify_new_comparewithold"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(8000 20000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



# actiontime=4
# action_name="2jump_step01verify_new"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 50000)
# hXarr=(0.1 0.1 0.1)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



# actiontime=5
# action_name="2jump_step005verify_new"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(20000 40000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=6
# action_name="2jump_step005verify_new"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



# actiontime=7
# action_name="2jump_step02verify_new_speedup"
# python_name="postdamage_2jump_smartx.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=8
# action_name="2jump_step01verify_new_speedup"
# python_name="postdamage_2jump_smartx.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.1 0.1 0.1)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# actiontime=9
# action_name="2jump_step005verify_new_speedup"
# python_name="postdamage_2jump_smartx.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0







# actiontime=10
# action_name="2jump_step02verify_cobweb"
# python_name="postdamage_2jump_cobweb.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=11
# action_name="2jump_step01verify_cobweb"
# python_name="postdamage_2jump_cobweb.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.1 0.1 0.1)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



# actiontime=12
# action_name="2jump_step005verify_cobweb"
# python_name="postdamage_2jump_cobweb.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(40000 60000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# actiontime=13
# action_name="2jump_step02verify_new"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(60000 80000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=14
# action_name="2jump_step005verify_new_try0005epfrac"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.005)
# fractionarr=(0.1 0.005)
# maxiterarr=(60000 80000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=15
# action_name="2jump_step02_cobwebnoHC"
# python_name="postdamage_2jump_cobweb_smartguess.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.01)
# fractionarr=(0.1 0.01)
# maxiterarr=(60000 80000)
# hXarr=(0.2 0.2 0.2)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# actiontime=16
# action_name="2jump_step005verify_new_try0005epfrac_morehours"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.005)
# fractionarr=(0.1 0.005)
# maxiterarr=(60000 80000)
# hXarr=(0.05 0.05 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# result: didn't even get into postdamage-tech2 case, something is off.

# actiontime=17
# action_name="2jump_step0.01verify_new_try0.001epfrac_morehours"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.001)
# fractionarr=(0.1 0.001)
# maxiterarr=(60000 80000)
# hXarr=(0.01 0.01 0.01)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


























# result is volatile, can't head towards conergence at 79999.
# but look back on actiontime=4, hey, 0.01 would cut it.
# so the smaller, is not always better?

# actiontime=30
# action_name="2jump_step0.10.10.1"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.001)
# fractionarr=(0.1 0.001)
# maxiterarr=(60000 80000)

# hXarr=(0.1 0.1 0.1)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0


# result is again, even more volatile,  0.001 is not working???

# actiontime=31
# action_name="2jump_step0.10.10.05"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.001)
# fractionarr=(0.1 0.001)
# maxiterarr=(60000 80000)

# hXarr=(0.1 0.1 0.05)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

# again, result is volatile and should take much longer time to approximate.

# actiontime=32
# action_name="2jump_step0.10.10.01"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.001)
# fractionarr=(0.1 0.001)
# maxiterarr=(60000 80000)

# hXarr=(0.1 0.1 0.01)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0



# actiontime=33
# action_name="2jump_step0.050.10.1"
# python_name="postdamage_2jump.py"
# NUM_DAMAGE=3
# ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
# epsilonarr=(0.1 0.001)
# fractionarr=(0.1 0.001)
# maxiterarr=(60000 80000)

# hXarr=(0.05 0.1 0.1)
# Xminarr=(4.00 0.0 -5.5 0.0)
# Xmaxarr=(9.00 4.0 0.0 3.0)
# xi_a=(1000.)
# xi_p=(1000.)
# psi0arr=(0.008 0.010 0.012)
# psi1arr=(0.8)
# LENGTH_xi=$((${#xi_a[@]}-1))
# count=0

actiontime=34
action_name="2jump_step0.010.10.1"
python_name="postdamage_2jump.py"
NUM_DAMAGE=3
ID_MAX_DAMAGE=$((NUM_DAMAGE-1))
epsilonarr=(0.1 0.001)
fractionarr=(0.1 0.001)
maxiterarr=(60000 80000)

hXarr=(0.01 0.1 0.1)
Xminarr=(4.00 0.0 -5.5 0.0)
Xmaxarr=(9.00 4.0 0.0 3.0)
xi_a=(1000.)
xi_p=(1000.)
psi0arr=(0.008 0.010 0.012)
psi1arr=(0.8)
LENGTH_xi=$((${#xi_a[@]}-1))
count=0





















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
#SBATCH --job-name=post${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.out
#SBATCH --error=./job-outs/${action_name}/xia_${xi_a[$j]}_xip_${xi_p[$j]}_PSI0_${PSI_0}_PSI1_${PSI_1}/mercury_post_$i.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/TwoCapital_Shrink/abatement/$python_name --num_gamma $NUM_DAMAGE --xi_a ${xi_a[$j]} --xi_g ${xi_p[$j]}  --epsilonarr ${epsilonarr[@]}  --fractionarr ${fractionarr[@]}   --maxiterarr ${maxiterarr[@]}  --id $i --psi_0 $PSI_0 --psi_1 $PSI_1 --name ${action_name} --hXarr ${hXarr[@]} --Xminarr ${Xminarr[@]} --Xmaxarr ${Xmaxarr[@]}

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
