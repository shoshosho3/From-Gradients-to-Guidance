#!/bin/bash

# --- Configuration ---
# Set the maximum number of jobs to run in parallel on the GPU
MAX_JOBS=3
CONDA_ENV_NAME="deepAL"
CONDA_EXECUTABLE=$(which conda)

# --- Sanity Check ---
if ! [ -x "$CONDA_EXECUTABLE" ]; then
    echo "ERROR: Conda executable not found. Please ensure conda is in your PATH."
    exit 1
fi

# Create a directory for log files and clean up old empty ones
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"
find "$LOG_DIR" -type f -size 0 -delete
echo "Cleaned up old empty log files in $LOG_DIR"

# --- Function to run a single experiment ---
# This function just defines ONE job. We will call it in a loop.
run_job() {
    strategy=$1
    dataset=$2
    quota=$3
    batch=$4
    initseed=$5
    extra_args_array=($6)

    log_name="${strategy}_${dataset}"
    if [[ "$6" == *"--REGL_factor 2"* ]]; then
        log_name+="_factor2"
    elif [[ "$6" == *"--REGL_factor 3"* ]]; then
        log_name+="_factor3"
    fi

    LOG_FILE="$LOG_DIR/${log_name}.log"

    echo "--- Queuing: ${log_name} ---"

    # Run the command in the background. The loop below will manage concurrency.
    "$CONDA_EXECUTABLE" run -n "$CONDA_ENV_NAME" python demo.py \
        --ALstrategy "$strategy" \
        --dataset_name "$dataset" \
        --quota "$quota" \
        --batch "$batch" \
        --initseed "$initseed" \
        --iteration 3 \
        "${extra_args_array[@]}" > "$LOG_FILE" 2>&1 &
}

# --- Main Job Queue and Management ---
echo "Starting experiment runner with a maximum of $MAX_JOBS parallel jobs."

# Define all the jobs you want to run
#run_job "RandomSampling" "CIFAR10" 10000 500 1000 ""
run_job "EGL"            "CIFAR10" 10000 500 1000 ""
run_job "LEGL"           "CIFAR10" 10000 500 1000 ""
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 2"
run_job "R-EGL"          "CIFAR10" 10000 500 1000 "--REGL_factor 3"
#run_job "RandomSampling" "FashionMNIST" 10000 250 500 ""
run_job "EGL"            "FashionMNIST" 10000 250 500 ""
run_job "LEGL"           "FashionMNIST" 10000 250 500 ""
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 2"
run_job "R-EGL"          "FashionMNIST" 10000 250 500 "--REGL_factor 3"

# --- THIS IS THE JOB MANAGEMENT LOOP ---
pids=()
# Get the process IDs of all background jobs started by this script
for job in $(jobs -p); do
    pids+=($job)
done

echo "Launched ${#pids[@]} jobs in the background. Now managing concurrency..."

# Loop while there are still jobs in our list
while (( ${#pids[@]} > 0 )); do
    # Check current number of running jobs that were started by this script
    running_jobs=0
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null; then
            ((running_jobs++))
        fi
    done

    # If we are at or above the job limit, wait for one to finish
    if (( running_jobs >= MAX_JOBS )); then
        # wait -n waits for the next background job to terminate
        wait -n
    fi

    # Prune the list of pids to only include those still running
    live_pids=()
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null; then
            live_pids+=($pid)
        fi
    done
    pids=("${live_pids[@]}")

    # Sleep for a moment to prevent this loop from consuming too much CPU
    sleep 1
done


echo "--- All experiments have finished. ---"