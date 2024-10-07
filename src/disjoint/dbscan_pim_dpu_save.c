#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 11
#endif

#define BLOCK_SIZE 256

typedef struct {
  float x[2];
  int cluster;
} DPUPoint;

typedef struct __attribute__((aligned(8))) {
  uint32_t n_points;
  float eps;
  int32_t min_pts;
  uint32_t padding; // Add padding to make the structure 16 bytes
} DBSCANParams;

__host DBSCANParams params;
__host DPUPoint points[BLOCK_SIZE];

BARRIER_INIT(my_barrier, NR_TASKLETS);

void dbscan_kernel(uint32_t tasklet_id) {
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&my_barrier);

  uint32_t nb_blocks = (params.n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float eps_squared = params.eps * params.eps;

  for (uint32_t block_id = 0; block_id < nb_blocks; block_id++) {
    uint32_t start_idx = block_id * BLOCK_SIZE;
    uint32_t end_idx = (start_idx + BLOCK_SIZE > params.n_points) ? params.n_points : start_idx + BLOCK_SIZE;
    uint32_t block_size = end_idx - start_idx;

    // Load block of points from MRAM to WRAM
    mram_read(&DPU_MRAM_HEAP_POINTER[start_idx * sizeof(DPUPoint)], points, block_size * sizeof(DPUPoint));

    // Process points in the block
    for (uint32_t i = tasklet_id; i < block_size; i += NR_TASKLETS) {
      if (points[i].cluster == -1) {
        int neighbors = 0;
        for (uint32_t j = 0; j < block_size; j++) {
          if (i != j) {
            float dx = points[i].x[0] - points[j].x[0];
            float dy = points[i].x[1] - points[j].x[1];
            float dist_squared = dx * dx + dy * dy;
            if (dist_squared <= eps_squared) {
              neighbors++;
            }
          }
        }

        if (neighbors >= params.min_pts) {
          points[i].cluster = block_id + 1; // Use block_id + 1 as temporary cluster ID
        }
      }
    }

    barrier_wait(&my_barrier);

    // Write updated points back to MRAM
    if (tasklet_id == 0) {
      mram_write(points, &DPU_MRAM_HEAP_POINTER[start_idx * sizeof(DPUPoint)], block_size * sizeof(DPUPoint));
    }
    barrier_wait(&my_barrier);
  }
}

int main() {
  uint32_t tasklet_id = me();

  if (tasklet_id == 0) {
    mem_reset();
  }

  dbscan_kernel(tasklet_id);

  return 0;
}