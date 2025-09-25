#!/bin/bash

# --- Configuration ---
# Run `which conda` to get this path.
CONDA_EXECUTABLE=$(which conda)

CONDA_ENV_NAME="deepAL"

# --- Sanity Check ---
if ! [ -x "$CONDA_EXECUTABLE" ]; then
    echo "ERROR: Conda executable not found. Please ensure conda is in your PATH."
    exit 1
fi

# Create a directory for log files if it doesn't exist
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"
# Clean up old empty logs before starting
find "$LOG_DIR" -type f -size 0 -delete

# --- Function to run a single experiment ---
run_job() {
    # $1: ALstrategy, $2: dataset_name, $3: quota, $4: batch, $5: initseed, $6: extra_args
    strategy=$1
    dataset=$2
    quota=$3
    batch=$4
    initseed=$5
    # Use an array for extra arguments to handle them safely
    extra_args_array=($6)

    log_name="${strategy}_${dataset}"
    if [[ "$6" == *"--REGL_factor 2"* ]]; then
        log_name+="_factor2"
    elif [[ "$6" == *"--REGL_factor 3"* ]]; then
        log_name+="_factor3"
    fi

    LOG_FILE="$LOG_DIR/${log_name}.log"

    echo "--- Starting: ${log_name} ---"
    echo "Logging output to: $LOG_FILE"

    # THE FIX IS HERE:
    # Instead of building a string, we call python directly and pass the arguments.
    # This is the robust way to do it. The shell handles each argument correctly.
    "$CONDA_EXECUTABLE" run -n "$CONDA_ENV_NAME" python demo.py \
        --ALstrategy "$strategy" \
        --dataset_name "$dataset" \
        --quota "$quota" \
        --batch "$batch" \
        --initseed "$initseed" \
        --iteration 3 \
        "${extra_args_array[@]}" > "$LOG_FILE" 2>&1 &
}

# --- Main Job Queue (same as before) ---
run_job "RandomSampling" "CIFAR10" 10000 500 1000 ""
run_job "EGL"            "CIFAR10" 10000 500 1000 ""
run_job "LEGL"           "CIFAR10" 10000 500 1000 ""
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 2"
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 3"
run_job "RandomSampling" "FashionMNIST" 10000 250 500 ""
run_job "EGL"            "FashionMNIST" 10000 250 500 ""
run_job "LEGL"           "FashionMNIST" 10000 250 500 ""
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 2"
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 3"

# Wait for all background jobs to complete
wait

echo "--- All experiments have finished. ---"