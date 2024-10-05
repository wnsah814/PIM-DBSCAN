#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define DIMENSIONS 2
#define UNCLASSIFIED -1
#define NOISE -2

typedef struct {
  double x[DIMENSIONS];
  int cluster;
} Point;

int *disjoint_set;
int set_size;

void init_disjoint_set(int size) {
  set_size = size;
  disjoint_set = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++) {
    disjoint_set[i] = i;
  }
}

void free_disjoint_set() { free(disjoint_set); }

int find(int x) {
  if (disjoint_set[x] != x) {
    disjoint_set[x] = find(disjoint_set[x]);
  }
  return disjoint_set[x];
}

void union_sets(int x, int y) {
  int xroot = find(x);
  int yroot = find(y);

  if (xroot != yroot) {
    disjoint_set[yroot] = xroot;
  }
}

static inline double squared_distance(const Point *a, const Point *b) {
  double sum = 0.0;
  for (int i = 0; i < DIMENSIONS; i++) {
    double diff = a->x[i] - b->x[i];
    sum += diff * diff;
  }
  return sum;
}

void dbscan(Point *points, int n_points, double eps, int min_pts) {
  init_disjoint_set(n_points);
  char *core_points = (char *)calloc(n_points, sizeof(char));
  int *neighbors = (int *)malloc(n_points * sizeof(int));
  double eps_squared = eps * eps;
  int cluster_id = 0;

  // First pass: Find core points and count neighbors
  for (int i = 0; i < n_points; i++) {
    int count = 0;
    Point *point_i = &points[i];
    for (int j = 0; j < n_points; j++) {
      if (i != j && squared_distance(point_i, &points[j]) <= eps_squared) {
        count++;
        if (count >= min_pts) {
          core_points[i] = 1;
          break;
        }
      }
    }
    neighbors[i] = count;
  }

  // Second pass: Merge clusters
  for (int i = 0; i < n_points; i++) {
    if (core_points[i]) {
      Point *point_i = &points[i];
      for (int j = i + 1; j < n_points; j++) {
        if (core_points[j] && squared_distance(point_i, &points[j]) <= eps_squared) {
          union_sets(i, j);
        }
      }
    }
  }

  // Assign cluster labels
  for (int i = 0; i < n_points; i++) {
    if (core_points[i]) {
      int root = find(i);
      if (points[root].cluster == UNCLASSIFIED) {
        points[root].cluster = cluster_id++;
      }
      points[i].cluster = points[root].cluster;
    } else if (neighbors[i] > 0) {
      points[i].cluster = UNCLASSIFIED; // Border point
    } else {
      points[i].cluster = NOISE;
    }
  }

  // Assign border points to clusters
  for (int i = 0; i < n_points; i++) {
    if (points[i].cluster == UNCLASSIFIED) {
      Point *point_i = &points[i];
      for (int j = 0; j < n_points; j++) {
        if (core_points[j] && squared_distance(point_i, &points[j]) <= eps_squared) {
          points[i].cluster = points[j].cluster;
          break;
        }
      }
    }
  }

  free(core_points);
  free(neighbors);
  free_disjoint_set();
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

  int n_points = 0;
  int max_points = 1000000;
  Point *points = (Point *)malloc(max_points * sizeof(Point));

  // Load data from CSV file
  FILE *file = fopen(data_file, "r");
  if (file == NULL) {
    printf("Error opening data file\n");
    return 1;
  }

  while (fscanf(file, "%lf,%lf", &points[n_points].x[0], &points[n_points].x[1]) == 2) {
    points[n_points].cluster = UNCLASSIFIED;
    n_points++;
    if (n_points >= max_points) {
      max_points *= 2;
      points = (Point *)realloc(points, max_points * sizeof(Point));
    }
  }
  fclose(file);

  // Load true labels (as float)
  double *labels_true = (double *)malloc(n_points * sizeof(double));
  file = fopen(labels_file, "r");
  if (file == NULL) {
    printf("Error opening labels file\n");
    free(points);
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

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);
  dbscan(points, n_points, eps, min_pts);
  gettimeofday(&end_time, NULL);

  double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

  // Prepare predicted labels
  int *labels_pred = (int *)malloc(n_points * sizeof(int));
  for (int i = 0; i < n_points; i++) {
    labels_pred[i] = points[i].cluster;
  }

  // Calculate accuracy (ARI)
  double ari = rand_index(labels_true, labels_pred, n_points);

  // 결과 파일 이름 생성
  char result_file[256];
  snprintf(result_file, sizeof(result_file), "%s_result.txt", output_prefix);

  // 결과 파일 열기
  FILE *result = fopen(result_file, "w");
  if (result == NULL) {
    printf("Error opening result file\n");
    return 1;
  }

  // 결과 출력 및 파일에 쓰기
  fprintf(result, "DBSCAN completed in %f seconds\n", time_taken);
  fprintf(result, "Adjusted Rand Index: %f\n", ari);

  // 첫 10개 포인트 결과 출력
  for (int i = 0; i < 10 && i < n_points; i++) {
    fprintf(result, "Point %d: true cluster %f, predicted cluster %d\n", i, labels_true[i], labels_pred[i]);
  }

  fclose(result);

  // 플롯 데이터 파일 이름 생성
  char plot_file[256];
  snprintf(plot_file, sizeof(plot_file), "%s_plot_data.csv", output_prefix);

  // 플롯 데이터 저장
  FILE *plot = fopen(plot_file, "w");
  if (plot == NULL) {
    printf("Error opening plot data file\n");
    free(points);
    free(labels_true);
    free(labels_pred);
    return 1;
  }

  fprintf(plot, "x,y,true_label,predicted_label\n");
  for (int i = 0; i < n_points; i++) {
    fprintf(plot, "%f,%f,%f,%d\n", points[i].x[0], points[i].x[1], labels_true[i], labels_pred[i]);
  }
  fclose(plot);

  printf("Results saved to %s\n", result_file);
  printf("Plot data saved to %s\n", plot_file);

  free(points);
  free(labels_true);
  free(labels_pred);

  return 0;
}