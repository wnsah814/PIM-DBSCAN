#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define DIMENSIONS 2
#define UNCLASSIFIED -1
#define NOISE -2
#define MAX_POINTS 65536

typedef struct {
  double x[DIMENSIONS];
  int cluster;
} Point;

int *disjoint_set;
int set_size;

void init_disjoint_set(int size) {
  set_size = size;
  disjoint_set = (int *)malloc(size * sizeof(int));
#pragma omp parallel for
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
#pragma omp critical
    {
      if (xroot < yroot) {
        disjoint_set[yroot] = xroot;
      } else {
        disjoint_set[xroot] = yroot;
      }
    }
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
#pragma omp parallel for
  for (int i = 0; i < n_points; i++) {
    int count = 0;
    for (int j = 0; j < n_points; j++) {
      if (i != j && squared_distance(&points[i], &points[j]) <= eps_squared) {
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
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_points; i++) {
    if (core_points[i]) {
      for (int j = i + 1; j < n_points; j++) {
        if (core_points[j] && squared_distance(&points[i], &points[j]) <= eps_squared) {
          union_sets(i, j);
        }
      }
    }
  }

// Assign cluster labels
#pragma omp parallel for
  for (int i = 0; i < n_points; i++) {
    if (core_points[i]) {
      int root = find(i);
#pragma omp critical
      {
        if (points[root].cluster == UNCLASSIFIED) {
          points[root].cluster = cluster_id++;
        }
      }
      points[i].cluster = points[root].cluster;
    } else if (neighbors[i] > 0) {
      points[i].cluster = UNCLASSIFIED; // Border point
    } else {
      points[i].cluster = NOISE;
    }
  }

// Assign border points to clusters
#pragma omp parallel for
  for (int i = 0; i < n_points; i++) {
    if (points[i].cluster == UNCLASSIFIED) {
      for (int j = 0; j < n_points; j++) {
        if (core_points[j] && squared_distance(&points[i], &points[j]) <= eps_squared) {
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
#pragma omp parallel for reduction(+ : tp, tn, fp, fn)
  for (int i = 0; i < n_points; i++) {
    for (int j = i + 1; j < n_points; j++) {
      if (fabs(labels_true[i] - labels_true[j]) < 1e-6 && labels_pred[i] == labels_pred[j])
        tp++;
      else if (fabs(labels_true[i] - labels_true[j]) >= 1e-6 && labels_pred[i] != labels_pred[j])
        tn++;
      else if (fabs(labels_true[i] - labels_true[j]) >= 1e-6 && labels_pred[i] == labels_pred[j])
        fp++;
      else if (fabs(labels_true[i] - labels_true[j]) < 1e-6 && labels_pred[i] != labels_pred[j])
        fn++;
    }
  }
  double ri = (double)(tp + tn) / (tp + tn + fp + fn);
  double expected_ri = ((double)(tp + fp) * (tp + fn) + (double)(tn + fp) * (tn + fn)) /
                       ((double)(tp + tn + fp + fn) * (tp + tn + fp + fn));
  return (ri - expected_ri) / (1 - expected_ri);
}

int main() {
  int n_points = 0;
  double eps = 1.6;
  int min_pts = 3;
  Point *points = (Point *)malloc(MAX_POINTS * sizeof(Point));

  // Load data from CSV file
  FILE *file = fopen("../data/dbscan_data_65536.csv", "r");
  if (file == NULL) {
    printf("Error opening data file\n");
    return 1;
  }

  while (fscanf(file, "%lf,%lf", &points[n_points].x[0], &points[n_points].x[1]) == 2) {
    points[n_points].cluster = UNCLASSIFIED;
    n_points++;
    if (n_points >= MAX_POINTS)
      break;
  }
  fclose(file);

  // Load true labels (as float)
  double *labels_true = (double *)malloc(n_points * sizeof(double));
  file = fopen("../data/dbscan_data_65536_labels.csv", "r");
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
  printf("Parallel DBSCAN completed in %f seconds\n", time_taken);

  // Prepare predicted labels
  int *labels_pred = (int *)malloc(n_points * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < n_points; i++) {
    labels_pred[i] = points[i].cluster;
  }

  // Calculate accuracy (ARI)
  double ari = rand_index(labels_true, labels_pred, n_points);
  printf("Adjusted Rand Index: %f\n", ari);

  // Print results (first 10 points)
  for (int i = 0; i < 10 && i < n_points; i++) {
    printf("Point %d: true cluster %f, predicted cluster %d\n", i, labels_true[i], labels_pred[i]);
  }

  // Save plot data
  file = fopen("dbscan_plot_data_65536.csv", "w");
  if (file == NULL) {
    printf("Error opening plot data file\n");
    free(points);
    free(labels_true);
    free(labels_pred);
    return 1;
  }

  fprintf(file, "x,y,true_label,predicted_label\n");
  for (int i = 0; i < n_points; i++) {
    fprintf(file, "%f,%f,%f,%d\n", points[i].x[0], points[i].x[1], labels_true[i], labels_pred[i]);
  }
  fclose(file);

  printf("Plot data saved to dbscan_plot_data_65536.csv\n");

  free(points);
  free(labels_true);
  free(labels_pred);

  return 0;
}