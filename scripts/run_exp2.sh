#!/bin/bash

BIN_DIR="./bin"
DATA_DIR="./data"
RESULTS_DIR="./exp2"

EPS=9
MIN_PTS=20

DATASETS=(
  # "blobs_16384_3clusters_2d"
  # "blobs_32768_3clusters_2d"
  # "blobs_65536_3clusters_2d" 
  "blobs_131072_3clusters_2d" # 1MB
  "blobs_262144_3clusters_2d"   # 2MB
  "blobs_524288_3clusters_2d"   # 4MB
  # "blobs_1048576_3clusters_2d"  # 8MB
  # "blobs_2097152_3clusters_2d"  #16MB # 64개를 사용했을 때만 구하면 됨
)

for DATASET in "${DATASETS[@]}"; do
  echo "Processing dataset: $DATASET"
  for dpus in 64 128 256 512 1024; do
    echo "Running with $dpus DPUs"
    sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BIN_DIR/dbscan_pim_host $DATA_DIR/${DATASET}.csv $EPS $MIN_PTS "${RESULTS_DIR}/pim_${DATASET}" ${dpus}
    sudo chown $USER:$USER "${RESULTS_DIR}/pim_${DATASET}_${dpus}_labels.txt"
    sudo chown $USER:$USER "${RESULTS_DIR}/pim_${DATASET}_${dpus}_result.txt"
    ari=$(python3 scripts/calculate_ari.py "$DATA_DIR/${DATASET}_labels.csv" "${RESULTS_DIR}/pim_${DATASET}_${dpus}_labels.txt")
    echo "PIM ARI: $ari" >> "${RESULTS_DIR}/pim_${DATASET}_${dpus}_result.txt"
  done
done