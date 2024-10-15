#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DPU_BINARY "./bin/dbscan_pim_dpu"

#define DIMENSIONS 2
#define UNCLASSIFIED -1
#define NOISE -2

#define DPU_AMOUNT 1024

typedef struct {
  int32_t x[DIMENSIONS];
  int32_t cluster;
  int32_t index;
} Point;

typedef struct {
  uint32_t *data;
  uint32_t size;
  uint32_t capacity;
} IntVector;

uint8_t *visited;

IntVector *create_int_vector(int initial_capacity) {
  IntVector *vec = (IntVector *)malloc(sizeof(IntVector));
  if (!vec)
    return NULL;
  vec->data = (uint32_t *)malloc(initial_capacity * sizeof(uint32_t));
  if (!vec->data) {
    free(vec);
    return NULL;
  }
  vec->size = 0;
  vec->capacity = initial_capacity;
  return vec;
}

void free_int_vector(IntVector *vec) {
  if (vec) {
    free(vec->data);
    free(vec);
  }
}

int push_back(IntVector *vec, uint32_t value) {
  if (vec->size == vec->capacity) {
    int new_capacity = vec->capacity * 2;
    uint32_t *new_data = (uint32_t *)realloc(vec->data, new_capacity * sizeof(uint32_t));
    if (!new_data)
      return 0;
    vec->data = new_data;
    vec->capacity = new_capacity;
  }
  vec->data[vec->size++] = value;
  return 1;
}

Point *points;
uint32_t nr_dpus;
uint32_t min_pts;

uint32_t load_data(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Error opening file");
    return -1;
  }

  uint32_t capacity = 65536;
  uint32_t count = 0;
  points = malloc(capacity * sizeof(Point));

  while (fscanf(file, "%d,%d", &points[count].x[0], &points[count].x[1]) == 2) {
    points[count].cluster = UNCLASSIFIED;
    points[count].index = count;
    count++;
    if (count >= capacity) {
      capacity *= 2;
      points = realloc(points, capacity * sizeof(Point));
    }
  }

  fclose(file);
  return count;
}

uint32_t get_neighbors_from_dpus(struct dpu_set_t set, const Point *query_point, uint32_t n_points,
                                 IntVector *neighbors) {
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_ASSERT(dpu_broadcast_to(set, "query_point", 0, query_point, sizeof(Point), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  uint32_t *counts = (uint32_t *)malloc(sizeof(uint32_t) * nr_dpus);
  uint32_t total_count = 0, max_count = 0;

  // WRAM을 통해 점의 이웃 개수를 먼저 받아온다.
  DPU_FOREACH(set, dpu, each_dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, &counts[each_dpu])); }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "neighbor_count", 0, 4, DPU_XFER_DEFAULT));

  for (uint32_t i = 0; i < nr_dpus; ++i) {
    total_count += counts[i];
    max_count = (counts[i] > max_count) ? counts[i] : max_count;
  }

  if (total_count < min_pts) {
    free(counts);
    return 0;
  }
  max_count = (max_count + 1) & ~(uint32_t)1;

  uint32_t *result = (uint32_t *)malloc(max_count * nr_dpus * sizeof(uint32_t));
  DPU_FOREACH(set, dpu, each_dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, &result[each_dpu * max_count])); }
  DPU_ASSERT(
      dpu_push_xfer(set, DPU_XFER_FROM_DPU, "mram_neighbors", 0, sizeof(uint32_t) * max_count, DPU_XFER_DEFAULT));

  for (uint32_t i = 0; i < nr_dpus; ++i) {
    for (uint32_t j = 0; j < counts[i]; ++j) {
      uint32_t idx = result[max_count * i + j];
      if (idx >= n_points)
        continue;
      if (!visited[idx]) {
        if (!push_back(neighbors, idx)) {
          printf("Failed to push neighbor\n");
          exit(1);
        }
        visited[idx] = 1;
      }
    }
  }
  free(result);
  free(counts);
  return total_count;
}

void expand_cluster(struct dpu_set_t set, int n_points, int point_id, int cluster_id, IntVector *neighbors,
                    IntVector *tmp_neighbors) {
  points[point_id].cluster = cluster_id;

  for (uint32_t i = 0; i < neighbors->size; ++i) {
    uint32_t current_point = neighbors->data[i];
    if (points[current_point].cluster == NOISE) {
      points[current_point].cluster = cluster_id;
    } else if (points[current_point].cluster == UNCLASSIFIED) {
      points[current_point].cluster = cluster_id;
      tmp_neighbors->size = 0;
      uint32_t neighbor_count = get_neighbors_from_dpus(set, &points[current_point], n_points, tmp_neighbors);
      if (neighbor_count < min_pts)
        continue;
      for (uint32_t j = 0; j < tmp_neighbors->size; ++j) {
        push_back(neighbors, tmp_neighbors->data[j]);
      }
    }
  }
}

void dbscan(struct dpu_set_t set, uint32_t n_points) {
  int cluster_id = 0;
  IntVector *neighbors = create_int_vector(n_points);
  IntVector *tmp_neighbors = create_int_vector(n_points);
  visited = (uint8_t *)calloc(n_points, sizeof(uint8_t));

  if (!neighbors || !tmp_neighbors || !visited) {
    fprintf(stderr, "Failed to allocate memory for neighbors\n");
    exit(1);
  }

  for (uint32_t i = 0; i < n_points; i++) {
    if (points[i].cluster != UNCLASSIFIED)
      continue;
    neighbors->size = 0;
    uint32_t neighbor_count = get_neighbors_from_dpus(set, &points[i], n_points, neighbors);

    if (neighbor_count < min_pts) {
      points[i].cluster = NOISE;
    } else {
      visited[i] = 1;
      cluster_id++;
      expand_cluster(set, n_points, i, cluster_id, neighbors, tmp_neighbors);
    }
  }
  free(neighbors);
  free(tmp_neighbors);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <data_file> <eps> <min_pts> <output_prefix>\n", argv[0]);
    return 1;
  }

  char *data_file = argv[1];
  int32_t eps = atoi(argv[2]);
  min_pts = atoi(argv[3]);
  char *output_prefix = argv[4];

  uint32_t n_points = load_data(data_file);
  if (n_points == 0) {
    return 1;
  }

  struct dpu_set_t set, dpu;

  DPU_ASSERT(dpu_alloc(DPU_AMOUNT, NULL, &set));
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  printf("Allocated %d DPU(s)\n", nr_dpus);

  uint32_t points_per_dpu = n_points / nr_dpus; // 나누어 떨어진다는 가정

  printf("points_per_dpu %u\n", points_per_dpu);

  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

  DPU_ASSERT(dpu_broadcast_to(set, "n_points", 0, &points_per_dpu, 4, DPU_XFER_DEFAULT));
  eps = eps * eps;
  DPU_ASSERT(dpu_broadcast_to(set, "eps_squared", 0, &eps, 4, DPU_XFER_DEFAULT));
  uint32_t each_dpu;
  DPU_FOREACH(set, dpu, each_dpu) {

    int start_idx = each_dpu * points_per_dpu;
    DPU_ASSERT(dpu_prepare_xfer(dpu, &points[start_idx]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "mram_points", 0, points_per_dpu * sizeof(Point), DPU_XFER_DEFAULT));

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  dbscan(set, n_points);

  gettimeofday(&end_time, NULL);
  double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

  printf("total time = %lf\n", time_taken);

  char result_file[256];
  snprintf(result_file, sizeof(result_file), "%s_result.txt", output_prefix);
  FILE *result = fopen(result_file, "w");
  if (result == NULL) {
    printf("Error opening result file\n");
    return 1;
  }
  fprintf(result, "DBSCAN completed in %f seconds\n", time_taken);
  fclose(result);

  char labels_output_file[256];
  snprintf(labels_output_file, sizeof(labels_output_file), "%s_labels.txt", output_prefix);
  FILE *labels_output = fopen(labels_output_file, "w");
  if (labels_output == NULL) {
    printf("Error opening labels output file\n");
    return 1;
  }
  for (uint32_t i = 0; i < n_points; i++) {
    fprintf(labels_output, "%d\n", points[i].cluster);
  }
  fclose(labels_output);

  printf("Results saved to %s\n", result_file);
  printf("Predicted labels saved to %s\n", labels_output_file);

  free(points);
  DPU_ASSERT(dpu_free(set));

  return 0;
}