#!/bin/sh
#SBATCH -J M1_H_sim 
#SBATCH -o  out.txt
#SBATCH -e  error.txt
#SBATCH -t 0-48:00:00  # days-hours:minutes

#SBATCH -N 1
#SBATCH -n 80 # used for MPI codes, otherwise leave at '1'
##SBATCH --ntasks-per-node=1  # don't trust SLURM to divide the cores evenly
##SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
##SBATCH --exclusive  # using MPI with 90+% of the cores you should go exclusive
#SBATCH --mem-per-cpu=2G  # memory per core; default is 1GB/core

## send mail to this address, alert at start, end and abortion of execution

START=$(date)
echo "Started running at $START."

export HDF5_USE_FILE_LOCKING=FALSE
unset DISPLAY

mpirun nrniv -mpi -python run_network.py config.json #srun

END=$(date)
echo "Done running simulation at $END"

TRIALNAME="baseline_135"
mkdir ../Analysis/simulation_results/"$TRIALNAME"
cp -a output/. ../Analysis/simulation_results/"$TRIALNAME"
cp -a ecp_tmp/. ../Analysis/simulation_results/"$TRIALNAME"
