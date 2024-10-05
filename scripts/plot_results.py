import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(data_file, result_file, output_file):
    # Read the plot data
    data = pd.read_csv(data_file)

    # Read the result file to get ARI and execution time
    with open(result_file, 'r') as f:
        lines = f.readlines()
        time_taken = float(lines[0].split()[-2])
        ari = float(lines[1].split()[-1])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot true labels
    sns.scatterplot(data=data, x='x', y='y', hue='true_label', palette='deep', ax=ax1)
    ax1.set_title('True Clusters')
    ax1.legend(title='True Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot predicted labels
    sns.scatterplot(data=data, x='x', y='y', hue='predicted_label', palette='deep', ax=ax2)
    ax2.set_title('DBSCAN Predicted Clusters')
    ax2.legend(title='Predicted Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add execution time and ARI to the plot
    plt.figtext(0.5, 0.01, f'Execution Time: {time_taken:.2f} seconds, ARI: {ari:.4f}', 
                ha='center', fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {output_file}")

def main():
    data_dir = './data'
    results_dir = './results'
    plots_dir = './plots'
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Plot results for each dataset and implementation
    for implementation in ['cpu', 'openmp', 'pim']:
        for data_file in os.listdir(data_dir):
            if data_file.endswith('.csv') and not data_file.endswith('_labels.csv'):
                dataset = data_file[:-4]  # Remove .csv extension
                data_path = os.path.join(data_dir, data_file)
                plot_data_file = os.path.join(results_dir, f"{implementation}_{dataset}_plot_data.csv")
                result_file = os.path.join(results_dir, f"{implementation}_{dataset}_result.txt")
                
                if os.path.exists(plot_data_file) and os.path.exists(result_file):
                    output_file = os.path.join(plots_dir, f'dbscan_results_{dataset}_{implementation}.png')
                    plot_results(plot_data_file, result_file, output_file)
                else:
                    print(f"Warning: Required files for {dataset} {implementation} not found. Skipping.")

if __name__ == "__main__":
    main()