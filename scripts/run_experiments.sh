#!/bin/bash

# 디렉토리 설정
BIN_DIR="./bin"
DATA_DIR="./data"
RESULTS_DIR="./results"

# 결과 디렉토리 생성
mkdir -p $RESULTS_DIR

# DBSCAN 파라미터
EPS=1.6
MIN_PTS=3

# 실행할 버전 설정
RUN_CPU=0
RUN_OPENMP=0
RUN_PIM=0

# 명령줄 인자 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--cpu) RUN_CPU=1 ;;
        -o|--openmp) RUN_OPENMP=1 ;;
        -p|--pim) RUN_PIM=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 아무 옵션도 선택되지 않았다면 모든 버전 실행
if [[ $RUN_CPU -eq 0 && $RUN_OPENMP -eq 0 && $RUN_PIM -eq 0 ]]; then
    RUN_CPU=1
    RUN_OPENMP=1
    RUN_PIM=1
fi

# data/ 디렉토리의 모든 CSV 파일에 대해 실험 실행
for data_file in $DATA_DIR/*.csv; do
    # 레이블 파일이 아닌 경우에만 처리
    if [[ $data_file != *"_labels.csv" ]]; then
        # 레이블 파일이 존재하는지 확인
        labels_file="${data_file%.*}_labels.csv"
        if [[ ! -f "$labels_file" ]]; then
            echo "Labels file not found for $data_file. Skipping."
            continue
        fi

        dataset=$(basename "$data_file" .csv)
        echo "Running experiments for $dataset dataset"
        
        if [[ $RUN_CPU -eq 1 ]]; then
            # CPU 버전
            echo "Running CPU version..."
            $BIN_DIR/dbscan_cpu "$data_file" "$labels_file" $EPS $MIN_PTS "$RESULTS_DIR/cpu_${dataset}"
        fi
        
        if [[ $RUN_OPENMP -eq 1 ]]; then
            # OpenMP 버전
            echo "Running OpenMP version..."
            $BIN_DIR/dbscan_cpu_openmp "$data_file" "$labels_file" $EPS $MIN_PTS "$RESULTS_DIR/openmp_${dataset}"
        fi
        
        if [[ $RUN_PIM -eq 1 ]]; then
            # PIM 버전
            echo "Running PIM version..."
            sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $BIN_DIR/dbscan_pim_host "$data_file" "$labels_file" $EPS $MIN_PTS "$RESULTS_DIR/pim_${dataset}"
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