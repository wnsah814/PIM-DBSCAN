import math
import sys
from typing import List, Tuple

Point = Tuple[float, float]

def squared_distance(a: Point, b: Point) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))

def region_query(points: List[Point], point_id: int, eps_squared: float) -> List[int]:
    return [i for i, point in enumerate(points) if squared_distance(points[point_id], point) <= eps_squared]

def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <data_file> <eps> <min_pts> <output_prefix>")
        sys.exit(1)

    data_file = sys.argv[1]
    eps = float(sys.argv[2])
    min_pts = int(sys.argv[3])
    output_prefix = sys.argv[4]

    # Load data from CSV file
    points = []
    with open(data_file, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            points.append((x, y))

    # Perform region query for the first point
    eps_squared = eps ** 2
    neighbors = region_query(points, 0, eps_squared)
    
    print(f"Number of neighbors for the first point: {len(neighbors)}")

    # Save the number of neighbors to a file
    result_file = f"{output_prefix}_neighbor_count.txt"
    with open(result_file, 'w') as f:
        f.write(f"Number of neighbors for the first point: {len(neighbors)}\n")

    print(f"Result saved to {result_file}")

if __name__ == "__main__":
    main()