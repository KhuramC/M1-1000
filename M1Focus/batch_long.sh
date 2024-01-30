#!/bin/bash

#SBATCH -J M1_H_long
#SBATCH -o  out_long.txt
#SBATCH -e  error_long.txt
#SBATCH -t 0-48:00:00  # days-hours:minutes


#SBATCH -N 1
#SBATCH -n 24 # used for MPI codes, otherwise leave at '1'
##SBATCH --ntasks-per-node=1  # don't trust SLURM to divide the cores evenly
##SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
##SBATCH --exclusive  # using MPI with 90+% of the cores you should go exclusive
#SBATCH --mem-per-cpu=2G  # memory per core; default is 1GB/core

START=$(date)
echo "Started running at $START."

mpirun nrniv -mpi -python run_network.py simulation_config_long.json #srun

END=$(date)
echo "Done running simulation at $END"
