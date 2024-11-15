CC = gcc
CFLAGS = -Wall -Wextra -std=c11
OMPFLAGS = -fopenmp
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

CPU_SRC = $(SRC_DIR)/dbscan_cpu.c
CPU_OMP_SRC = $(SRC_DIR)/dbscan_cpu_openmp.c
PIM_HOST_SRC = $(SRC_DIR)/dbscan_pim_host.c
PIM_DPU_SRC = $(SRC_DIR)/dbscan_pim_dpu.c

CPU_TARGET = $(BIN_DIR)/dbscan_cpu
CPU_OMP_TARGET = $(BIN_DIR)/dbscan_cpu_openmp
PIM_HOST_TARGET = $(BIN_DIR)/dbscan_pim_host
PIM_DPU_TARGET = $(BIN_DIR)/dbscan_pim_dpu

# DPU 컴파일러 및 플래그
DPU_CC = dpu-upmem-dpurte-clang
DPU_CFLAGS = -DNR_TASKLETS=11 -DSTACK_SIZE_DEFAULT=4096 -I/home/wnsah814/upmem-sdk/include/dpu

# 호스트 PIM 컴파일 플래그
# PIM_HOST_CFLAGS = $(shell dpu-pkg-config --cflags)
# PIM_HOST_LIBS = $(shell dpu-pkg-config --libs dpu)

# 기본 타겟 설정
TARGETS = $(CPU_TARGET)

# OpenMP 버전 컴파일 여부
ifeq ($(OPENMP),1)
    TARGETS += $(CPU_OMP_TARGET)
endif

# PIM 버전 컴파일 여부
ifeq ($(PIM),1)
    TARGETS += $(PIM_HOST_TARGET) $(PIM_DPU_TARGET)
endif

all: create_dirs $(TARGETS)

create_dirs:
	@mkdir -p $(BIN_DIR)
	@mkdir -p data
	@mkdir -p results
	@mkdir -p plots

$(CPU_TARGET): $(CPU_SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(CPU_OMP_TARGET): $(CPU_OMP_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

$(PIM_HOST_TARGET): $(PIM_HOST_SRC)
	$(CC) $(CFLAGS) $< -o $@ `dpu-pkg-config --cflags --libs dpu`

$(PIM_DPU_TARGET): $(PIM_DPU_SRC)
	$(DPU_CC) $(DPU_CFLAGS) $< -o $@

clean:
	rm -f $(BIN_DIR)/*

.PHONY: all clean create_dirs