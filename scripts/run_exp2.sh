#!/bin/bash

BIN_DIR="./bin"
DATA_DIR="./data"
RESULTS_DIR="./exp2"

EPS=50
MIN_PTS=20
# DATASET="blobs_65536_3clusters_2d"
# DATASET="blobs_131072_3clusters_2d"
DATASET="blobs_262144_3clusters_2d"

for dpus in 64 128 256 512 1024; do
  sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BIN_DIR/dbscan_pim_host $DATA_DIR/${DATASET}.csv $EPS $MIN_PTS "${RESULTS_DIR}/pim_${DATASET}" ${dpus}
  sudo chown $USER:$USER "${RESULTS_DIR}/pim_${DATASET}_${dpus}_labels.txt"
  sudo chown $USER:$USER "${RESULTS_DIR}/pim_${DATASET}_${dpus}_result.txt"
  ari=$(python3 scripts/calculate_ari.py "$DATA_DIR/${DATASET}_labels.csv" "${RESULTS_DIR}/pim_${DATASET}_${dpus}_labels.txt")
  echo "PIM ARI: $ari" >> "${RESULTS_DIR}/pim_${DATASET}_${dpus}_result.txt"
done