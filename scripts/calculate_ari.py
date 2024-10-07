import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score

def calculate_ari(true_labels_file, pred_labels_file):
    true_labels = np.loadtxt(true_labels_file)
    pred_labels = np.loadtxt(pred_labels_file)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_ari.py <true_labels_file> <pred_labels_file>")
        sys.exit(1)

    true_labels_file = sys.argv[1]
    pred_labels_file = sys.argv[2]
    
    ari = calculate_ari(true_labels_file, pred_labels_file)
    print(f"{ari}")