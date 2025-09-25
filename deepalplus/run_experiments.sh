#!/bin/bash

# --- Configuration ---

# creating a directory for log files if it doesn't exist
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"

# --- Function to run a single experiment ---
run_job() {
    # $1: ALstrategy, $2: dataset_name, $3: quota, $4: batch, $5: initseed, $6: extra_args
    strategy=$1
    dataset=$2
    quota=$3
    batch=$4
    initseed=$5
    extra_args=$6

    # creating a clean name for the log file
    log_name="${strategy}_${dataset}"
    if [[ "$extra_args" == *"--REGL_factor 2"* ]]; then
        log_name+="_factor2"
    elif [[ "$extra_args" == *"--REGL_factor 3"* ]]; then
        log_name+="_factor3"
    fi

    LOG_FILE="$LOG_DIR/${log_name}.log"

    echo "--- Starting: ${log_name} ---"
    echo "Full command: python demo.py --ALstrategy $strategy --dataset_name $dataset --quota $quota --batch $batch --initseed $initseed --iteration 3 $extra_args"
    echo "Logging output to: $LOG_FILE"

    # Execute the command in the background, redirecting stdout and stderr to the log file
    python demo.py --ALstrategy "$strategy" --dataset_name "$dataset" --quota "$quota" --batch "$batch" --initseed "$initseed" --iteration 3 $extra_args > "$LOG_FILE" 2>&1 &
}

# --- Main Job Queue ---

# --- CIFAR10 Runs ---
run_job "RandomSampling" "CIFAR10" 10000 500 1000 ""
run_job "EGL"            "CIFAR10" 10000 500 1000 ""
run_job "LEGL"           "CIFAR10" 10000 500 1000 ""
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 2"
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 3"

# --- FashionMNIST Runs ---
run_job "RandomSampling" "FashionMNIST" 10000 250 500 ""
run_job "EGL"            "FashionMNIST" 10000 250 500 ""
run_job "LEGL"           "FashionMNIST" 10000 250 500 ""
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 2"
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 3"

# Wait for all remaining background jobs to complete
wait

echo "--- All experiments have finished. ---"