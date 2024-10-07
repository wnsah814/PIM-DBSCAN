#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define UNCLASSIFIED -1
#define NOISE -2
#define DIMENSIONS 2

typedef struct {
  double x[DIMENSIONS];
  int cluster;
} Point;

typedef struct {
  int *data;
  int size;
  int capacity;
} IntVector;

static inline double squared_distance(const Point *a, const Point *b) {
  double sum = 0.0;
  for (int i = 0; i < DIMENSIONS; i++) {
    double diff = a->x[i] - b->x[i];
    sum += diff * diff;
  }
  return sum;
}

IntVector *create_int_vector(int initial_capacity) {
  IntVector *vec = (IntVector *)malloc(sizeof(IntVector));
  if (!vec)
    return NULL;
  vec->data = (int *)malloc(initial_capacity * sizeof(int));
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

int push_back(IntVector *vec, int value) {
  if (vec->size == vec->capacity) {
    int new_capacity = vec->capacity * 2;
    int *new_data = (int *)realloc(vec->data, new_capacity * sizeof(int));
    if (!new_data)
      return 0;
    vec->data = new_data;
    vec->capacity = new_capacity;
  }
  vec->data[vec->size++] = value;
  return 1;
}

void region_query(const Point *points, int n_points, int point_id, double eps_squared, IntVector *neighbors) {
  neighbors->size = 0;
  for (int i = 0; i < n_points; i++) {
    if (squared_distance(&points[point_id], &points[i]) <= eps_squared) {
      if (!push_back(neighbors, i)) {
        fprintf(stderr, "Failed to add neighbor\n");
        exit(1);
      }
    }
  }
}

void expand_cluster(Point *points, int n_points, int point_id, int cluster_id, double eps_squared, int min_pts,
                    IntVector *neighbors, IntVector *new_neighbors) {
  points[point_id].cluster = cluster_id;

  for (int i = 0; i < neighbors->size; i++) {
    int current_point = neighbors->data[i];
    if (points[current_point].cluster == UNCLASSIFIED) {
      points[current_point].cluster = cluster_id;
      new_neighbors->size = 0;
      region_query(points, n_points, current_point, eps_squared, new_neighbors);

      if (new_neighbors->size >= min_pts) {
        for (int j = 0; j < new_neighbors->size; j++) {
          if (points[new_neighbors->data[j]].cluster == UNCLASSIFIED ||
              points[new_neighbors->data[j]].cluster == NOISE) {
            if (!push_back(neighbors, new_neighbors->data[j])) {
              fprintf(stderr, "Failed to add neighbor\n");
              exit(1);
            }
          }
        }
      }
    }
  }
}

void dbscan(Point *points, int n_points, double eps, int min_pts) {
  int cluster_id = 0;
  double eps_squared = eps * eps;
  IntVector *neighbors = create_int_vector(n_points);
  IntVector *new_neighbors = create_int_vector(n_points);
  if (!neighbors || !new_neighbors) {
    fprintf(stderr, "Failed to allocate memory for neighbors\n");
    exit(1);
  }

  for (int i = 0; i < n_points; i++) {
    if (points[i].cluster != UNCLASSIFIED)
      continue;

    region_query(points, n_points, i, eps_squared, neighbors);

    if (neighbors->size < min_pts) {
      points[i].cluster = NOISE;
    } else {
      cluster_id++;
      expand_cluster(points, n_points, i, cluster_id, eps_squared, min_pts, neighbors, new_neighbors);
    }
  }

  free_int_vector(neighbors);
  free_int_vector(new_neighbors);
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

  // Create result file name
  char result_file[256];
  snprintf(result_file, sizeof(result_file), "%s_result.txt", output_prefix);

  // Open result file
  FILE *result = fopen(result_file, "w");
  if (result == NULL) {
    printf("Error opening result file\n");
    return 1;
  }

  // Write results to file
  fprintf(result, "DBSCAN completed in %f seconds\n", time_taken);
  fprintf(result, "Adjusted Rand Index: %f\n", ari);

  // Print first 10 point results
  for (int i = 0; i < 10 && i < n_points; i++) {
    fprintf(result, "Point %d: true cluster %f, predicted cluster %d\n", i, labels_true[i], labels_pred[i]);
  }

  fclose(result);

  // Create plot data file name
  char plot_file[256];
  snprintf(plot_file, sizeof(plot_file), "%s_plot_data.csv", output_prefix);

  // Save plot data
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