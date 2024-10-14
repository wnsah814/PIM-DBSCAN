#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdio.h>

#define DIMENSIONS 2
#define MAX_NEIGHBORS 1000000 // MRAM이 받올 수 있는 최대 점의 개수
#define CACHE_SIZE 80         // WRAM이 가져올 수 있는 최대 점의 개수
#ifndef NR_TASKLETS           // 오류 안뜨게 하는 용도
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

__mram_noinit uint64_t mram_n_points;
__mram_noinit int64_t mram_eps_squared;
__mram_noinit Point mram_points[MAX_NEIGHBORS];
__mram_noinit Point mram_query_point;

// host에게 전달할 값들
__mram_noinit Point mram_neighbors[MAX_NEIGHBORS];
__host uint32_t neighbor_count; // host는 WRAM에 접근해 이 값을 가져온다?

// tasklet_0 에 의해 변경될 값들
__dma_aligned uint32_t n_points;
__dma_aligned int32_t eps_squared;
__dma_aligned Point query_point;

__dma_aligned uint32_t points_per_tasklet;

__dma_aligned Point output_buffer[2][CACHE_SIZE];
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
  // mram에서 가져온 점들을 저장하는 곳
  // tasklet마다 고유한 stack 공간
  __dma_aligned Point point_cache[CACHE_SIZE];

  uint32_t tasklet_id = me();

  if (tasklet_id == 0) {
    neighbor_count = 0; // 필요한가? WRAM이라면 초기화될 것 같고, mram의 데이터라면 해줘야하고
    active_buffer = 0;
    buffer_index = 0;
    mram_read(&mram_n_points, &n_points, 8);
    mram_read(&mram_eps_squared, &eps_squared, 8);
    mram_read(&mram_query_point, &query_point, sizeof(Point));

    // uint32_t max_points_per_tasklet = (1 << 11) / sizeof(Point);
    points_per_tasklet = n_points / NR_TASKLETS;
    if (points_per_tasklet > CACHE_SIZE) {
      points_per_tasklet = CACHE_SIZE;
    }
  }

  barrier_wait(&setup_barrier);

  // printf("%u %lf %u %lf %lf\n", n_points, eps_squared, points_per_tasklet, query_point.x[0], query_point.x[1]);

  for (int i = tasklet_id * points_per_tasklet; i < n_points; i += points_per_tasklet * NR_TASKLETS) {
    uint32_t cache_size = (i + points_per_tasklet > n_points) ? (n_points - i) : points_per_tasklet;
    /*
    NOTE. cache_size가 8의 배수여야하는 줄 알았지만, sizeof(Point)에 8을 포함해서 자연스럽게 배수가 됨.
    */
    mram_read(&mram_points[i], point_cache, cache_size * sizeof(Point));
    for (int j = 0; j < cache_size; ++j) {
      if (squared_distance(point_cache[j].x, query_point.x) <= eps_squared) {
        mutex_lock(neighbor_mutex);

        output_buffer[active_buffer][buffer_index] = point_cache[j];
        buffer_index++;
        neighbor_count++;

        if (buffer_index == CACHE_SIZE || neighbor_count == n_points) {
          uint32_t current_buffer_size = buffer_index;
          uint32_t current_neighbor_count = neighbor_count - current_buffer_size;

          mutex_lock(buffer_mutex);

          // Switch buffers
          active_buffer = 1 - active_buffer;
          buffer_index = 0;
          mutex_unlock(neighbor_mutex);

          // Write current buffer to MRAM
          mram_write(output_buffer[1 - active_buffer], &mram_neighbors[current_neighbor_count],
                     sizeof(Point) * current_buffer_size);

          mutex_unlock(buffer_mutex);
        } else {
          mutex_unlock(neighbor_mutex);
        }
      }
    }
  }
  barrier_wait(&final_sync_barrier);

  // 버퍼에 남아있는 점들
  if (tasklet_id == 0 && buffer_index > 0) {
    mram_write(output_buffer[active_buffer], &mram_neighbors[neighbor_count - buffer_index],
               sizeof(Point) * buffer_index);
  }

  return 0;
}