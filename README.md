# DBSCAN Implementation Comparison

This project compares different implementations of the DBSCAN algorithm: CPU, OpenMP, and PIM (Processing-in-Memory).

## Directory Structure

```
project_root/
├── src/                 # Source code for DBSCAN implementations
│   ├── dbscan_cpu.c
│   ├── dbscan_cpu_openmp.c
│   ├── dbscan_pim_host.c
│   └── dbscan_pim_dpu.c
├── bin/                 # Compiled binaries
├── data/                # Input data files and corresponding label files
├── results/             # Experiment results and output files
├── plots/               # Generated plot images
├── scripts/             # Scripts for running experiments, plotting results, and generating datasets
│   ├── run_experiments.sh
│   ├── plot_results.py
│   ├── generate_dataset.py
│   └── calculate_ari.py
├── Makefile             # Makefile for compiling the project
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Requirements

- GCC compiler
- OpenMP
- UPMEM SDK (for PIM implementation)
- Python 3.6+
- Python libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/wnsah814/PIM-DBSCAN.git
   cd PIM-DBSCAN
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have GCC, OpenMP, and UPMEM SDK installed on your system.

## Usage

1. Generate dataset:
   ```
   python3 scripts/generate_dataset.py
   ```
   Follow the prompts to create your dataset. The generated data and corresponding labels will be saved in the `data/` directory.

2. Compile the project:
   - Compile all versions:
     ```
     make OPENMP=1 PIM=1
     ```
   - Compile only CPU version:
     ```
     make
     ```
   - Compile with OpenMP:
     ```
     make OPENMP=1
     ```
   - Compile with PIM:
     ```
     make PIM=1
     ```

3. Run experiments:
   - Run all versions:
     ```
     ./scripts/run_experiments.sh
     ```
   - Run only CPU version:
     ```
     ./scripts/run_experiments.sh -c
     ```
   - Run OpenMP version:
     ```
     ./scripts/run_experiments.sh -o
     ```
   - Run PIM version:
     ```
     ./scripts/run_experiments.sh -p
     ```
   - Run multiple versions:
     ```
     ./scripts/run_experiments.sh -c -o
     ```

4. View results in the `results/` directory and plots in the `plots/` directory.

## Key Changes in This Version

1. C implementations (CPU, OpenMP, PIM) no longer process label files. They only handle input data and output predicted labels.
2. A new Python script `calculate_ari.py` has been added to calculate the Adjusted Rand Index (ARI) using scikit-learn.
3. The `run_experiments.sh` script now coordinates the execution of C programs and the ARI calculation script.
4. Results now include execution time from C programs and ARI calculated by the Python script.

## Customization

- Modify `scripts/generate_dataset.py` to change dataset generation parameters.
- Adjust `EPS` and `MIN_PTS` values in `scripts/run_experiments.sh` to change DBSCAN parameters.
- Modify `scripts/plot_results.py` for different plotting options.

## Notes

- The PIM version requires sudo privileges to run.
- Ensure all necessary libraries and paths are correctly set up before running the PIM version.
- Make sure to generate or place your dataset and corresponding labels in the `data/` directory before running experiments.
- The Makefile will automatically create necessary directories if they don't exist.

## Troubleshooting

If you encounter any issues:
1. Ensure all required dependencies are installed.
2. Check that the data files and corresponding label files are in the correct format and location.
3. Verify that you have the necessary permissions to run the PIM version.
4. If you encounter Python-related errors, ensure you've installed all required libraries using `pip install -r requirements.txt`.

For any persistent problems, please open an issue in the project repository.