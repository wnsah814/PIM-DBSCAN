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
// #define DPU_AMOUNT 64

// typedef struct {
//   double x[DIMENSIONS];
//   union {
//     uint64_t label;
//     uint64_t index;
//   };
// } Point;

typedef struct {
  double x[DIMENSIONS];
  int32_t cluster;
  int32_t index;
} Point;

Point *points;

typedef struct {
  Point *neighbors;
  int neighbor_count;
} dpu_result_t;

int load_data(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Error opening file");
    return -1;
  }

  int capacity = 65536;
  int count = 0;
  points = malloc(capacity * sizeof(Point));

  while (fscanf(file, "%lf,%lf", &points[count].x[0], &points[count].x[1]) == 2) {
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
/*
// DPU에서 이웃 점들을 가져오는 함수
void get_neighbors_from_dpus(struct dpu_set_t set, Point *query_point, double eps, Point *all_points, int n_points,
                             Point **neighbors, int *neighbor_count) {
  // uint32_t nr_dpus;
  // DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

  // query point 전송
  printf("sending point (%lf, %lf)\n", points[0].x[0], points[0].x[1]);
  DPU_ASSERT(dpu_broadcast_to(set, "mram_query_point", 0, &points[0], sizeof(Point), DPU_XFER_DEFAULT));

  // DPU 실행
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  // 결과 수집
  *neighbor_count = 0;
  *neighbors = malloc(n_points * sizeof(Point)); // 최대 가능한 크기로 할당

  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_FOREACH(set, dpu, each_dpu) {
    int dpu_neighbor_count;
    DPU_ASSERT(dpu_copy_from(dpu, "neighbor_count", 0, &dpu_neighbor_count, sizeof(int)));

    if (dpu_neighbor_count > 0) {
      DPU_ASSERT(dpu_copy_from(dpu, "neighbors", 0, *neighbors + *neighbor_count, dpu_neighbor_count * sizeof(Point)));
      *neighbor_count += dpu_neighbor_count;
    }
  }

  // 메모리 재할당 (필요한 만큼만)
  *neighbors = realloc(*neighbors, *neighbor_count * sizeof(Point));
}

// DBSCAN 알고리즘
void dbscan(struct dpu_set_t set, int n_points, double eps, int min_pts) {
  int cluster_id = 0;

  for (int i = 0; i < n_points; i++) {
    if (points[i].cluster != UNCLASSIFIED)
      continue;

    Point *neighbors;
    int neighbor_count;
    get_neighbors_from_dpus(set, &points[i], eps, points, n_points, &neighbors, &neighbor_count);

    if (neighbor_count < min_pts) {
      points[i].cluster = NOISE;
    } else {
      cluster_id++;
      points[i].cluster = cluster_id;

      for (int j = 0; j < neighbor_count; j++) {
        int neighbor_index = points[j].cluster; // cluster 필드를 인덱스로 사용
        if (points[neighbor_index].cluster == UNCLASSIFIED || points[neighbor_index].cluster == NOISE) {
          if (points[neighbor_index].cluster == UNCLASSIFIED) {
            Point *sub_neighbors;
            int sub_neighbor_count;
            get_neighbors_from_dpus(set, &points[neighbor_index], eps, points, n_points, &sub_neighbors,
                                    &sub_neighbor_count);

            if (sub_neighbor_count >= min_pts) {
              for (int k = 0; k < sub_neighbor_count; k++) {
                int sub_index = sub_neighbors[k].cluster;
                if (points[sub_index].cluster == UNCLASSIFIED) {
                  neighbors[neighbor_count++] = points[sub_index];
                }
              }
            }
            free(sub_neighbors);
          }
          points[neighbor_index].cluster = cluster_id;
        }
      }
    }
    free(neighbors);
  }
}
*/
void dump_mram(struct dpu_set_t dpu, const char *symbol_name, size_t offset, size_t size) {
  uint8_t *buffer = (uint8_t *)malloc(size);
  if (!buffer) {
    fprintf(stderr, "Memory allocation failed\n");
    return;
  }

  DPU_ASSERT(dpu_copy_from(dpu, symbol_name, offset, buffer, size));

  printf("MRAM Dump of %s (offset: %zu, size: %zu bytes):\n", symbol_name, offset, size);
  for (size_t i = 0; i < size; i++) {
    if (i % 16 == 0) {
      printf("\n%04zx: ", i);
    }
    printf("%02x ", buffer[i]);
  }
  printf("\n");

  free(buffer);
}

void test_first_point(struct dpu_set_t set, uint32_t nr_dpus) {
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  uint32_t nums[nr_dpus];

  printf("sending point (%lf, %lf)\n", points[0].x[0], points[0].x[1]);
  DPU_ASSERT(dpu_broadcast_to(set, "mram_query_point", 0, &points[0], sizeof(Point), DPU_XFER_DEFAULT));

  printf("starting the program...\n");
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  printf("gathering the number of neighbors...\n");
  uint32_t total_nums = 0;
  DPU_FOREACH(set, dpu, each_dpu) { DPU_ASSERT(dpu_prepare_xfer(dpu, &nums[each_dpu])); }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "neighbor_count", 0, 4, DPU_XFER_DEFAULT));
  DPU_FOREACH(set, dpu, each_dpu) { total_nums += nums[each_dpu]; }
  printf("total sum = %u\n", total_nums);
}

void test_dump(struct dpu_set_t set, uint32_t nr_dpus, uint32_t points_per_dpu) {
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  uint32_t nums[nr_dpus];

  printf("sending point (%lf, %lf)\n", points[0].x[0], points[0].x[1]);
  DPU_ASSERT(dpu_broadcast_to(set, "mram_query_point", 0, &points[0], sizeof(Point), DPU_XFER_DEFAULT));

  printf("starting the program...\n");
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  // MRAM 덤프 수행
  DPU_FOREACH(set, dpu) {
    dump_mram(dpu, "mram_points", 0, points_per_dpu * sizeof(Point));
    break; // 첫 번째 DPU만 덤프
  }
}

void print_test(struct dpu_set_t set, uint32_t nr_dpus) {
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  uint32_t ids[nr_dpus];
  printf("\n##LOG##\n");

  printf("sending point (%lf, %lf)\n", points[0].x[0], points[0].x[1]);
  DPU_ASSERT(dpu_broadcast_to(set, "mram_query_point", 0, &points[0], sizeof(Point), DPU_XFER_DEFAULT));

  printf("starting the program...\n");
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  printf("gathering the output...\n");
  DPU_FOREACH(set, dpu) { DPU_ASSERT(dpu_log_read(dpu, stdout)); }

  printf("##done##\n");
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <data_file> <eps> <min_pts> <output_prefix>\n", argv[0]);
    return 1;
  }

  char *data_file = argv[1];
  double eps = atof(argv[2]);
  int min_pts = atoi(argv[3]);
  char *output_prefix = argv[4];

  int n_points = load_data(data_file);
  if (n_points < 0) {
    return 1;
  }

  // for (int i = 0; i < 10; ++i) {
  //   printf("%lf, %lf\n", points[i].x[0], points[i].x[1]);
  // }

  struct dpu_set_t set, dpu;
  uint32_t nr_dpus;

  DPU_ASSERT(dpu_alloc(DPU_AMOUNT, NULL, &set));
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  printf("Allocated %d DPU(s)\n", nr_dpus);

  // 각 DPU에 할당할 포인트 수 계산
  // uint64_t points_per_dpu = (n_points + nr_dpus - 1) / nr_dpus;
  uint32_t points_per_dpu = n_points / nr_dpus; // 나누어 떨어진다는 가정

  printf("points_per_dpu %u\n", points_per_dpu);

  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

  DPU_ASSERT(dpu_broadcast_to(set, "mram_n_points", 0, &points_per_dpu, 8, DPU_XFER_DEFAULT));
  eps = eps * eps;
  DPU_ASSERT(dpu_broadcast_to(set, "mram_eps_squared", 0, &eps, sizeof(eps), DPU_XFER_DEFAULT));
  uint32_t each_dpu;
  DPU_FOREACH(set, dpu, each_dpu) {

    int start_idx = each_dpu * points_per_dpu;
    DPU_ASSERT(dpu_prepare_xfer(dpu, &points[start_idx]));
    // int end_idx = (each_dpu + 1) * points_per_dpu;
    // if (end_idx > n_points)
    //   end_idx = n_points;
    // 나누어 떨어진다는 가정

    // DPU_ASSERT(dpu_copy_to(dpu, "my_dpu_id", 0, &each_dpu, 8));
    // DPU_ASSERT(dpu_prepare_xfer(dpu, &points[each_dpu * points_per_dpu]));

    // DPU_ASSERT(dpu_copy_to(dpu, "points", 0, points + start_idx, (end_idx - start_idx) * sizeof(Point)));
    // DPU_ASSERT(dpu_copy_to(dpu, "n_points", 0, &points_per_dpu, sizeof(uint64_t)));
  }
  // DPU_ASSERT(dpu_push_xfer(dpu, DPU_XFER_TO_DPU, "mram_points", 0, points_per_dpu, DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "mram_points", 0, points_per_dpu * sizeof(Point), DPU_XFER_DEFAULT));

  // 타이머 시작
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  // print_test(set, nr_dpus);
  test_first_point(set, nr_dpus);
  // test_dump(set, nr_dpus, points_per_dpu);
  // DBSCAN 실행
  // dbscan(set, n_points, eps, min_pts);

  // 타이머 종료
  gettimeofday(&end_time, NULL);
  double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

  printf("total time = %lf\n", time_taken);
  // 결과 저장
  /*
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
    for (int i = 0; i < n_points; i++) {
      fprintf(labels_output, "%d\n", points[i].cluster);
    }
    fclose(labels_output);

    printf("Results saved to %s\n", result_file);
    printf("Predicted labels saved to %s\n", labels_output_file);
  */
  // 메모리 해제
  free(points);
  DPU_ASSERT(dpu_free(set));

  return 0;
}