#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DPU_BINARY "./bin/dbscan_pim_dpu"
#define MAX_DPU_COUNT 1024

// Host-side structures
typedef struct {
  double x[2];
  int cluster;
} Point;

typedef struct __attribute__((aligned(8))) {
  uint32_t n_points;
  float eps;
  int32_t min_pts;
  uint32_t padding; // Add padding to make the structure 16 bytes
} DBSCANParams;

// DPU-side structures (defined in both host and DPU programs)
typedef struct {
  float x[2];
  int cluster;
} DPUPoint;

// Function prototypes
void init_points(DPUPoint *dpu_points, Point *points, uint32_t n_points);
void dbscan_host(struct dpu_set_t set, DBSCANParams *params, DPUPoint *dpu_points, uint32_t n_points);
double rand_index(double *labels_true, int *labels_pred, int n_points);

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
  int n_points = 0;
  int max_points = 1000000;
  Point *points = (Point *)malloc(max_points * sizeof(Point));
  double *labels_true = (double *)malloc(max_points * sizeof(double));

  FILE *file = fopen(data_file, "r");
  if (file == NULL) {
    printf("Error opening data file\n");
    return 1;
  }

  while (fscanf(file, "%lf,%lf", &points[n_points].x[0], &points[n_points].x[1]) == 2) {
    points[n_points].cluster = -1;
    n_points++;
    if (n_points >= max_points) {
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

  for (int i = 0; i < n_points; i++) {
    if (fscanf(file, "%lf", &labels_true[i]) != 1) {
      printf("Error reading labels\n");
      fclose(file);
      free(points);
      free(labels_true);
      return 1;
    }
  }
  fclose(file);

  // Initialize DPUs
  struct dpu_set_t set, dpu;
  uint32_t nr_dpus;

  DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  printf("Allocated %d DPU(s)\n", nr_dpus);

  // Prepare DPU points
  uint32_t points_per_dpu = (n_points + nr_dpus - 1) / nr_dpus;
  DPUPoint *dpu_points = (DPUPoint *)malloc(n_points * sizeof(DPUPoint));
  init_points(dpu_points, points, n_points);

  // Prepare DBSCAN parameters
  DBSCANParams params = {.n_points = n_points, .eps = eps, .min_pts = min_pts};

  // Run DBSCAN
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  dbscan_host(set, &params, dpu_points, n_points);

  gettimeofday(&end_time, NULL);
  double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

  // Prepare predicted labels
  int *labels_pred = (int *)malloc(n_points * sizeof(int));
  for (int i = 0; i < n_points; i++) {
    labels_pred[i] = dpu_points[i].cluster;
  }

  // Calculate accuracy (ARI)
  double ari = rand_index(labels_true, labels_pred, n_points);

  // Save results
  char result_file[256];
  snprintf(result_file, sizeof(result_file), "%s_result.txt", output_prefix);
  FILE *result = fopen(result_file, "w");
  if (result == NULL) {
    printf("Error opening result file\n");
    return 1;
  }

  fprintf(result, "DBSCAN completed in %f seconds\n", time_taken);
  fprintf(result, "Adjusted Rand Index: %f\n", ari);

  for (int i = 0; i < 10 && i < n_points; i++) {
    fprintf(result, "Point %d: true cluster %f, predicted cluster %d\n", i, labels_true[i], labels_pred[i]);
  }

  fclose(result);

  // Save plot data
  char plot_file[256];
  snprintf(plot_file, sizeof(plot_file), "%s_plot_data.csv", output_prefix);
  FILE *plot = fopen(plot_file, "w");
  if (plot == NULL) {
    printf("Error opening plot data file\n");
    return 1;
  }

  fprintf(plot, "x,y,true_label,predicted_label\n");
  for (int i = 0; i < n_points; i++) {
    fprintf(plot, "%f,%f,%f,%d\n", dpu_points[i].x[0], dpu_points[i].x[1], labels_true[i], labels_pred[i]);
  }
  fclose(plot);

  printf("Results saved to %s\n", result_file);
  printf("Plot data saved to %s\n", plot_file);

  // Clean up
  free(points);
  free(labels_true);
  free(labels_pred);
  free(dpu_points);
  DPU_ASSERT(dpu_free(set));

  return 0;
}

void init_points(DPUPoint *dpu_points, Point *points, uint32_t n_points) {
  for (uint32_t i = 0; i < n_points; i++) {
    dpu_points[i].x[0] = (float)points[i].x[0];
    dpu_points[i].x[1] = (float)points[i].x[1];
    dpu_points[i].cluster = -1;
  }
}

void dbscan_host(struct dpu_set_t set, DBSCANParams *params, DPUPoint *dpu_points, uint32_t n_points) {
  uint32_t nr_dpus;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  uint32_t points_per_dpu = (n_points + nr_dpus - 1) / nr_dpus;

  // Transfer data to DPUs
  printf("Transferring parameters to DPUs. Size of DBSCANParams: %zu\n", sizeof(DBSCANParams));
  DPU_ASSERT(dpu_broadcast_to(set, "params", 0, params, sizeof(DBSCANParams), DPU_XFER_DEFAULT));

  struct dpu_set_t dpu;
  uint32_t dpu_id = 0;
  DPU_FOREACH(set, dpu, dpu_id) {
    uint32_t start = dpu_id * points_per_dpu;
    uint32_t end = (dpu_id + 1) * points_per_dpu;
    if (end > n_points)
      end = n_points;
    uint32_t dpu_n_points = end - start;

    DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_points[start]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "points", 0, sizeof(DPUPoint) * points_per_dpu, DPU_XFER_DEFAULT));

  // Run DBSCAN on DPUs
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  // Retrieve results from DPUs
  DPU_FOREACH(set, dpu, dpu_id) {
    uint32_t start = dpu_id * points_per_dpu;
    uint32_t end = (dpu_id + 1) * points_per_dpu;
    if (end > n_points)
      end = n_points;
    uint32_t dpu_n_points = end - start;

    DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_points[start]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "points", 0, sizeof(DPUPoint) * points_per_dpu, DPU_XFER_DEFAULT));

  // Merge clusters
  int max_cluster = 0;
  for (uint32_t i = 0; i < n_points; i++) {
    if (dpu_points[i].cluster > max_cluster) {
      max_cluster = dpu_points[i].cluster;
    }
  }

  int *cluster_map = (int *)malloc((max_cluster + 1) * sizeof(int));
  for (int i = 0; i <= max_cluster; i++) {
    cluster_map[i] = i;
  }

  for (uint32_t i = 0; i < n_points; i++) {
    for (uint32_t j = i + 1; j < n_points; j++) {
      if (dpu_points[i].cluster != -1 && dpu_points[j].cluster != -1 &&
          dpu_points[i].cluster != dpu_points[j].cluster) {
        float dx = dpu_points[i].x[0] - dpu_points[j].x[0];
        float dy = dpu_points[i].x[1] - dpu_points[j].x[1];
        if (dx * dx + dy * dy <= params->eps * params->eps) {
          int root_i = cluster_map[dpu_points[i].cluster];
          int root_j = cluster_map[dpu_points[j].cluster];
          if (root_i < root_j) {
            cluster_map[root_j] = root_i;
          } else {
            cluster_map[root_i] = root_j;
          }
        }
      }
    }
  }

  // Apply merged clusters
  for (uint32_t i = 0; i < n_points; i++) {
    if (dpu_points[i].cluster != -1) {
      dpu_points[i].cluster = cluster_map[dpu_points[i].cluster];
    }
  }

  free(cluster_map);
}

double rand_index(double *labels_true, int *labels_pred, int n_points) {
  long long tp = 0, tn = 0, fp = 0, fn = 0;
  for (int i = 0; i < n_points; i++) {
    double label_true_i = labels_true[i];
    int label_pred_i = labels_pred[i];
    for (int j = i + 1; j < n_points; j++) {
      if (fabs(label_true_i - labels_true[j]) < 1e-6 && label_pred_i == labels_pred[j])
        tp++;
      else if (fabs(label_true_i - labels_true[j]) >= 1e-6 && label_pred_i != labels_pred[j])
        tn++;
      else if (fabs(label_true_i - labels_true[j]) >= 1e-6 && label_pred_i == labels_pred[j])
        fp++;
      else if (fabs(label_true_i - labels_true[j]) < 1e-6 && label_pred_i != labels_pred[j])
        fn++;
    }
  }
  double ri = (double)(tp + tn) / (tp + tn + fp + fn);
  double expected_ri = ((double)(tp + fp) * (tp + fn) + (double)(tn + fp) * (tn + fn)) /
                       ((double)(tp + tn + fp + fn) * (tp + tn + fp + fn));
  return (ri - expected_ri) / (1 - expected_ri);
}