# --- Configuration ---
strategies=(
    "RandomSampling" "VanillaEGL" "AdvancedEGL" "AdvancedLEGL"
    "DiversityEGL" "DiversityLEGL" "LeastConfidence" "EntropySampling"
)

datasets=("CIFAR10" "MNIST" "FashionMNIST" "SVHN")

# --- Execution ---
# Loop through the requested seeds
for seed in {42..44}; do
    echo "========================================="
    echo "        STARTING RUN FOR SEED: $seed       "
    echo "========================================="

    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Set dataset-specific parameters
        case "$dataset" in
            "CIFAR10")
                batch_size=500
                init_seed_size=1000
                ;;
            *) # For all other datasets (MNIST, FashionMNIST, SVHN)
                batch_size=250
                init_seed_size=500
                ;;
        esac

        echo "-----------------------------------------"
        echo "Running Dataset: $dataset (Batch: $batch_size, InitSeed: $init_seed_size)"
        echo "-----------------------------------------"

        # Loop through each strategy
        for strat in "${strategies[@]}"; do
            echo "--> Running Strategy: $strat"
            python demo.py \
                --ALstrategy "$strat" \
                --dataset_name "$dataset" \
                --quota 5000 \
                --batch "$batch_size" \
                --initseed "$init_seed_size" \
                --iteration 1 \
                --seed "$seed"
        done
    done
done

echo "All runs completed successfully."