#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdio.h>

#define DIMENSIONS 2
#define MAX_NEIGHBORS 2048 // MRAM이 받올 수 있는 최대 점의 개수
#define CACHE_SIZE 123     // WRAM이 가져올 수 있는 최대 점의 개수
#define BUFFER_SIZE 512    // 결과 저장 버퍼
#ifndef NR_TASKLETS        // 오류 안뜨게 하는 용도
#define NR_TASKLETS 11
#endif

typedef struct {
  int32_t x[DIMENSIONS];
  int32_t cluster;
  int32_t index;
} Point;

BARRIER_INIT(setup_barrier, NR_TASKLETS);
BARRIER_INIT(final_sync_barrier, NR_TASKLETS);
MUTEX_INIT(neighbor_mutex);
MUTEX_INIT(buffer_mutex);

__mram_noinit Point mram_points[MAX_NEIGHBORS];

// host에게 전달할 값들
__mram_noinit uint32_t mram_neighbors[MAX_NEIGHBORS];
__host uint32_t neighbor_count;

__host uint32_t n_points;
__host int32_t eps_squared;
__host Point query_point;

__dma_aligned uint32_t points_per_tasklet;
__dma_aligned uint32_t output_buffer[2][BUFFER_SIZE];
__dma_aligned uint8_t active_buffer; // 0 or 1
__dma_aligned uint32_t buffer_index;

int32_t squared_distance(const int32_t *a, const int32_t *b) {
  int32_t sum = 0;
  for (int i = 0; i < DIMENSIONS; i++) {
    int32_t diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

int main() {
  __dma_aligned Point point_cache[CACHE_SIZE];

  uint32_t tasklet_id = me();

  if (tasklet_id == 0) {
    neighbor_count = 0;
    active_buffer = 0;
    buffer_index = 0;

    // uint32_t max_points_per_tasklet = (1 << 11) / sizeof(Point);
    points_per_tasklet = n_points / NR_TASKLETS;
    if (points_per_tasklet > CACHE_SIZE) {
      points_per_tasklet = CACHE_SIZE;
    }
  }

  barrier_wait(&setup_barrier);

  for (int i = tasklet_id * points_per_tasklet; i < n_points; i += points_per_tasklet * NR_TASKLETS) {
    uint32_t cache_size = (i + points_per_tasklet > n_points) ? (n_points - i) : points_per_tasklet;
    mram_read(&mram_points[i], point_cache, cache_size * sizeof(Point));
    for (int j = 0; j < cache_size; ++j) {
      if (squared_distance(point_cache[j].x, query_point.x) <= eps_squared) {
        mutex_lock(neighbor_mutex);
        output_buffer[active_buffer][buffer_index] = (uint32_t)point_cache[j].index;
        buffer_index++;
        neighbor_count++;

        if (buffer_index == BUFFER_SIZE || neighbor_count == n_points) {
          uint32_t current_buffer_size = buffer_index;
          uint32_t current_neighbor_count = neighbor_count - current_buffer_size;
          mutex_lock(buffer_mutex);
          active_buffer = 1 - active_buffer;
          buffer_index = 0;
          mutex_unlock(neighbor_mutex);
          mram_write(output_buffer[1 - active_buffer], &mram_neighbors[current_neighbor_count],
                     sizeof(uint32_t) * current_buffer_size);
          mutex_unlock(buffer_mutex);
        } else {
          mutex_unlock(neighbor_mutex);
        }
      }
    }
  }
  barrier_wait(&final_sync_barrier);

  if (tasklet_id == 0 && buffer_index > 0) {
    mram_write(output_buffer[active_buffer], &mram_neighbors[neighbor_count - buffer_index],
               sizeof(uint32_t) * (buffer_index + buffer_index % 2));
  }
  return 0;
}