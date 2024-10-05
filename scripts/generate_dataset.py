import os
from sklearn.datasets import make_moons, make_blobs, make_circles
import numpy as np
import matplotlib.pyplot as plt

def get_user_input(prompt, default_value, value_type=int):
    user_input = input(f"{prompt} (default: {default_value}): ")
    if user_input == "":
        return default_value
    return value_type(user_input)

def generate_custom_dataset(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    return X, y

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def main():
    # Get folder path from user
    default_folder = "./data"
    folder_path = input(f"Enter folder path to save datasets (default: {default_folder}): ").strip()
    if folder_path == "":
        folder_path = default_folder
    
    create_folder_if_not_exists(folder_path)

    print("Choose a dataset type:")
    print("1. Blobs")
    print("2. Moons")
    print("3. Circles")
    print("4. Custom (random Gaussian)")
    
    dataset_type = get_user_input("Enter your choice", 1)

    # Get common user inputs
    n_samples = get_user_input("Enter number of samples", 65536)
    random_state = get_user_input("Enter random state", 42)

    if dataset_type == 1:  # Blobs
        n_centers = get_user_input("Enter number of clusters", 3)
        n_features = get_user_input("Enter number of features", 2)
        cluster_std = get_user_input("Enter cluster standard deviation", 1.0, float)
        X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, 
                          cluster_std=cluster_std, random_state=random_state)
        dataset_name = f"blobs_{n_samples}_{n_centers}clusters_{n_features}d"
    
    elif dataset_type == 2:  # Moons
        noise = get_user_input("Enter noise level", 0.1, float)
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        dataset_name = f"moons_{n_samples}_{noise}noise"
    
    elif dataset_type == 3:  # Circles
        noise = get_user_input("Enter noise level", 0.1, float)
        factor = get_user_input("Enter factor (separation between circles)", 0.8, float)
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
        dataset_name = f"circles_{n_samples}_{noise}noise_{factor}factor"
    
    elif dataset_type == 4:  # Custom
        n_features = get_user_input("Enter number of features", 2)
        X, y = generate_custom_dataset(n_samples, n_features)
        dataset_name = f"custom_{n_samples}_{n_features}d"
    
    else:
        print("Invalid choice. Exiting.")
        return

    # Plot a subset of the data
    max_plot_samples = min(1000, n_samples)
    plt.figure(figsize=(10, 8))
    if X.shape[1] == 2:
        plt.scatter(X[:max_plot_samples, 0], X[:max_plot_samples, 1], c=y[:max_plot_samples])
    elif X.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:max_plot_samples, 0], X[:max_plot_samples, 1], X[:max_plot_samples, 2], c=y[:max_plot_samples])
    else:
        print("Can only plot 2D or 3D data. Skipping plot for higher dimensions.")
        
    plt.title(f'Synthetic Dataset for DBSCAN ({dataset_name})')
    plot_path = os.path.join(folder_path, f'{dataset_name}.png')
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")

    # Save the dataset to a CSV file
    data_path = os.path.join(folder_path, f'{dataset_name}.csv')
    np.savetxt(data_path, X, delimiter=',')
    print(f"Dataset saved as {data_path}")

    # Save the labels separately
    labels_path = os.path.join(folder_path, f'{dataset_name}_labels.csv')
    np.savetxt(labels_path, y, delimiter=',')
    print(f"Labels saved as {labels_path}")

if __name__ == "__main__":
    main()