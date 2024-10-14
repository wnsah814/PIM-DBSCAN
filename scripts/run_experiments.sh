#!/bin/bash

# Directory setup
BIN_DIR="./bin"
DATA_DIR="./data"
RESULTS_DIR="./results"

# Create results directory
mkdir -p $RESULTS_DIR

# DBSCAN parameters
EPS=50
MIN_PTS=20

# Set versions to run
RUN_CPU=0
RUN_OPENMP=0
RUN_PIM=0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--cpu) RUN_CPU=1 ;;
        -o|--openmp) RUN_OPENMP=1 ;;
        -p|--pim) RUN_PIM=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# If no options selected, run all versions
if [[ $RUN_CPU -eq 0 && $RUN_OPENMP -eq 0 && $RUN_PIM -eq 0 ]]; then
    RUN_CPU=1
    RUN_OPENMP=1
    RUN_PIM=1
fi

# Run experiments for all CSV files in data/ directory
for data_file in $DATA_DIR/*.csv; do
    # Process only if not a labels file
    if [[ $data_file != *"_labels.csv" ]]; then
        # Check if labels file exists
        labels_file="${data_file%.*}_labels.csv"
        if [[ ! -f "$labels_file" ]]; then
            echo "Labels file not found for $data_file. Skipping."
            continue
        fi

        dataset=$(basename "$data_file" .csv)
        echo "Running experiments for $dataset dataset"
        
        if [[ $RUN_CPU -eq 1 ]]; then
            # CPU version
            echo "Running CPU version..."
            $BIN_DIR/dbscan_cpu "$data_file" $EPS $MIN_PTS "$RESULTS_DIR/cpu_${dataset}"
            # Calculate ARI
            ari=$(python3 scripts/calculate_ari.py "$labels_file" "${RESULTS_DIR}/cpu_${dataset}_labels.txt")
            echo "CPU ARI: $ari" >> "${RESULTS_DIR}/cpu_${dataset}_result.txt"
        fi
        
        if [[ $RUN_OPENMP -eq 1 ]]; then
            # OpenMP version
            echo "Running OpenMP version..."
            $BIN_DIR/dbscan_cpu_openmp "$data_file" $EPS $MIN_PTS "$RESULTS_DIR/openmp_${dataset}"
            # Calculate ARI
            ari=$(python3 scripts/calculate_ari.py "$labels_file" "${RESULTS_DIR}/openmp_${dataset}_labels.txt")
            echo "OpenMP ARI: $ari" >> "${RESULTS_DIR}/openmp_${dataset}_result.txt"
        fi
        
        if [[ $RUN_PIM -eq 1 ]]; then
            # PIM version
            echo "Running PIM version..."
            sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BIN_DIR/dbscan_pim_host "$data_file" $EPS $MIN_PTS "$RESULTS_DIR/pim_${dataset}"
            
            # Change permissions of the output files
            sudo chown $USER:$USER "${RESULTS_DIR}/pim_${dataset}_labels.txt"
            sudo chown $USER:$USER "${RESULTS_DIR}/pim_${dataset}_result.txt"
            
            # Calculate ARI
            ari=$(python3 scripts/calculate_ari.py "$labels_file" "${RESULTS_DIR}/pim_${dataset}_labels.txt")
            echo "PIM ARI: $ari" >> "${RESULTS_DIR}/pim_${dataset}_result.txt"
        fi
        
        echo "Finished experiments for $dataset dataset"
        echo "----------------------------------------"
    fi
done

echo "All experiments completed. Results are saved in $RESULTS_DIR"

# Plot results
echo "Plotting results..."
python3 scripts/plot_results.py

echo "Experiment and plotting completed."