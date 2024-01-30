#!/bin/bash


RUN_NUM="${RUN_NUM:-1}"
echo $RUN_NUM

OUTPUT_DIR=../Analysis/simulation_results_v2/baseline/baseline_$RUN_NUM sbatch batch_baseline.sh
OUTPUT_DIR=../Analysis/simulation_results_v2/short/short_$RUN_NUM sbatch batch_short.sh
OUTPUT_DIR=../Analysis/simulation_results_v2/long/long_$RUN_NUM sbatch batch_long.sh
#OUTPUT_DIR=./outout/baseline/run$RUN_NUM sbatch batch_baseline.sh
#OUTPUT_DIR=./outout/short/run$RUN_NUM sbatch batch_short.sh
#OUTPUT_DIR=./outout/long/run$RUN_NUM sbatch batch_long.sh