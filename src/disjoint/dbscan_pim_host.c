#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DIMENSIONS 2
#define DPU_BINARY "./bin/dbscan_pim_dpu"
#define DEBUG

typedef struct {
  double x[DIMENSIONS];
} Point;

int *disjoint_set; // to merge centroids
int nr_points;

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: %s <data_file> <labels_file> <eps> <min_pts> <output_prefix>\n", argv[0]);
    return 1;
  }

  char *data_file = argv[1];
  char *labels_file = argv[2];
  double eps = atof(argv[3]);
  int min_pts = atoi(argv[4]);
  char *output_prefix = argv[5];

  // Load data and labels
  nr_points = 0;
  int max_points = 1000000;
  Point *points = (Point *)malloc(max_points * sizeof(Point));
  double *labels_true = (double *)malloc(max_points * sizeof(double));

  FILE *file = fopen(data_file, "r");
  if (file == NULL) {
    printf("Error opening data file\n");
    return 1;
  }

  // Suppose 2 dimension

  while (fscanf(file, "%lf,%lf", &points[nr_points].x[0], &points[nr_points].x[1]) == 2) {
    nr_points++;
    if (nr_points >= max_points) {
      max_points *= 2;
      points = (Point *)realloc(points, max_points * sizeof(Point));
      labels_true = (double *)realloc(labels_true, max_points * sizeof(double));
    }
  }
  fclose(file);

  file = fopen(labels_file, "r");
  if (file == NULL) {
    printf("Error opening labels file\n");
    free(points);
    free(labels_true);
    return 1;
  }

  for (int i = 0; i < nr_points; i++) {
    if (fscanf(file, "%lf", &labels_true[i]) != 1) {
      printf("Error reading labels\n");
      fclose(file);
      free(points);
      free(labels_true);
      return 1;
    }
  }
  fclose(file);

  disjoint_set = (int *)malloc(sizeof(int) * nr_points);
  for (int i = 0; i < nr_points; ++i) {
    disjoint_set[i] = i;
  }

  struct dpu_set_t set, dpu;
  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

#ifdef DEBUG
  uint32_t nr_dpus;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  printf("Allocated %d DPU(s)\n", nr_dpus);
#endif

  return 0;
}