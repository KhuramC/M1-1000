#!/bin/bash


# Usage: RUNS=10 ./batch_run_multiple.sh 
# Will kick off 10 runs for each 



RUNS="${RUNS:-1}"

for ((i = 1 ; i <= $RUNS ; i++)); do
	RUN_NUM=$i ./batch_all.sh
done