#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define UNCLASSIFIED -1
#define NOISE -2
#define DIMENSIONS 2

typedef struct {
  int32_t x[DIMENSIONS];
  int32_t cluster;
} Point;

typedef struct {
  uint32_t *data;
  int size;
  int capacity;
} IntVector;

uint8_t *visited; // Global visited array

static inline uint32_t squared_distance(const Point *a, const Point *b) {
  uint32_t sum = 0;
  for (int i = 0; i < DIMENSIONS; i++) {
    int32_t diff = a->x[i] - b->x[i];
    sum += diff * diff;
  }
  return sum;
}

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
    printf("set capacity to %d\n", new_capacity);
    uint32_t *new_data = (uint32_t *)realloc(vec->data, new_capacity * sizeof(uint32_t));
    if (!new_data)
      return 0;
    vec->data = new_data;
    vec->capacity = new_capacity;
  }
  vec->data[vec->size++] = value;
  return 1;
}

int region_query(const Point *points, int n_points, int point_id, uint32_t eps_squared, IntVector *neighbors) {
  int neighbor_count = 0;
  neighbors->size = 0;

  for (int i = 0; i < n_points; i++) {
    if (squared_distance(&points[point_id], &points[i]) <= eps_squared) {
      neighbor_count++;
      if (!visited[i]) {
        if (!push_back(neighbors, i)) {
          fprintf(stderr, "Failed to add neighbor in region_query\n");
          exit(1);
        }
        visited[i] = 1; // Mark as visited to avoid duplicate additions
      }
    }
  }
  return neighbor_count;
}

void expand_cluster(Point *points, int n_points, int point_id, int cluster_id, uint32_t eps_squared, int min_pts,
                    IntVector *neighbors) {
  points[point_id].cluster = cluster_id;

  for (int i = 0; i < neighbors->size; i++) {
    int current_point = neighbors->data[i];
    if (points[current_point].cluster == UNCLASSIFIED || points[current_point].cluster == NOISE) {
      if (points[current_point].cluster == UNCLASSIFIED) {
        IntVector *new_neighbors = create_int_vector(n_points);
        int neighbor_count = region_query(points, n_points, current_point, eps_squared, new_neighbors);

        if (neighbor_count >= min_pts) {
          for (int j = 0; j < new_neighbors->size; j++) {
            // if (!visited[new_neighbors->data[j]]) {
            if (!push_back(neighbors, new_neighbors->data[j])) {
              fprintf(stderr, "Failed to add neighbor in expand_cluster\n");
              exit(1);
            }
            // visited[new_neighbors->data[j]] = 1;
            // }
          }
        }
        free_int_vector(new_neighbors);
      }
      points[current_point].cluster = cluster_id;
    }
  }
}

void dbscan(Point *points, int n_points, uint32_t eps, int min_pts) {
  int cluster_id = 0;
  uint32_t eps_squared = eps * eps;
  IntVector *neighbors = create_int_vector(n_points);
  visited = (uint8_t *)calloc(n_points, sizeof(uint8_t));

  if (!neighbors || !visited) {
    fprintf(stderr, "Failed to allocate memory\n");
    exit(1);
  }

  for (int i = 0; i < n_points; i++) {
    if (points[i].cluster != UNCLASSIFIED)
      continue;

    // memset(visited, 0, n_points * sizeof(uint8_t)); // Reset visited array
    neighbors->size = 0; // Clear neighbors
    int neighbor_count = region_query(points, n_points, i, eps_squared, neighbors);

    if (neighbor_count < min_pts) {
      points[i].cluster = NOISE;
    } else {
      visited[i] = 1;
      cluster_id++;
      expand_cluster(points, n_points, i, cluster_id, eps_squared, min_pts, neighbors);
    }
  }

  free_int_vector(neighbors);
  free(visited);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <data_file> <eps> <min_pts> <output_prefix>\n", argv[0]);
    return 1;
  }

  char *data_file = argv[1];
  uint32_t eps = atoi(argv[2]);
  int min_pts = atoi(argv[3]);
  char *output_prefix = argv[4];

  int n_points = 0;
  int max_points = 1000000;
  Point *points = (Point *)malloc(max_points * sizeof(Point));

  // Load data from CSV file
  FILE *file = fopen(data_file, "r");
  if (file == NULL) {
    printf("Error opening data file\n");
    return 1;
  }

  while (fscanf(file, "%d,%d", &points[n_points].x[0], &points[n_points].x[1]) == 2) {
    points[n_points].cluster = UNCLASSIFIED;
    n_points++;
    if (n_points >= max_points) {
      max_points *= 2;
      points = (Point *)realloc(points, max_points * sizeof(Point));
    }
  }
  fclose(file);

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);
  dbscan(points, n_points, eps, min_pts);
  gettimeofday(&end_time, NULL);

  double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;

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

  fclose(result);

  // Create labels file name
  char labels_output_file[256];
  snprintf(labels_output_file, sizeof(labels_output_file), "%s_labels.txt", output_prefix);

  // Save predicted labels
  FILE *labels_output = fopen(labels_output_file, "w");
  if (labels_output == NULL) {
    printf("Error opening labels output file\n");
    free(points);
    return 1;
  }

  for (int i = 0; i < n_points; i++) {
    fprintf(labels_output, "%d\n", points[i].cluster);
  }
  fclose(labels_output);

  printf("Results saved to %s\n", result_file);
  printf("Predicted labels saved to %s\n", labels_output_file);

  free(points);

  return 0;
}