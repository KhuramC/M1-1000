#!/bin/bash


# Usage: START=1 RUNS=10 ./batch_run_multiple.sh 
# Will kick off 10 runs for each starting at #1.

START="${START:-1}"

RUNS="${RUNS:-1}"

for ((i = $START ; i < $RUNS+$START ; i++)); do
	RUN_NUM=$i ./batch_all.sh
done


