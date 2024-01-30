#!/bin/sh
#SBATCH -J M1_sim 
#SBATCH -o  ./stdout/M1_sim.o%j.out
#SBATCH -e  ./stdout/M1_sim.e%j.error
#SBATCH -t 0-48:00:00  # days-hours:minutes

#SBATCH -N 1
#SBATCH -n 50 # used for MPI codes, otherwise leave at '1'
##SBATCH --ntasks-per-node=1  # don't trust SLURM to divide the cores evenly
##SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
##SBATCH --exclusive  # using MPI with 90+% of the cores you should go exclusive
#SBATCH --mem-per-cpu=2G  # memory per core; default is 1GB/core

## send mail to this address, alert at start, end and abortion of execution
##SBATCH --mail-type=ALL
##SBATCH --mail-user=kac2cf@umsystem.edu

START=$(date)
echo "Started running at $START."

export HDF5_USE_FILE_LOCKING=FALSE
unset DISPLAY

mpirun ./components/mechanisms/x86_64/special -mpi -python run_network.py config_no_STP.json True # args: config file, whether use coreneuron

END=$(date)
echo "Done running simulation at $END"

TRIALNAME="baseline_r13_no_STP"
mkdir ../Analysis/simulation_results/"$TRIALNAME"
cp -a output/. ../Analysis/simulation_results/"$TRIALNAME"
cp -a ecp_tmp/. ../Analysis/simulation_results/"$TRIALNAME"
