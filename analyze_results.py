import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE_PART_NUM = 6  # expected number of parts in the filename when split by '_'
DATASET_INDEX = 0
METHOD_INDEX = 1
SEED_INDEX = 5


def trapezoid_area(data_points):
    """
    Calculate the area under the curve using the trapezoidal rule.
    :param data_points: list of (x, y) tuples
    :return: float area
    """

    actual_area = 0.0
    # using the trapezoidal rule to calculate the area under the curve
    for i in range(len(data_points) - 1):
        x1, y1 = data_points[i]
        x2, y2 = data_points[i + 1]

        # area of one trapezoid: (average height) * width
        trapezoid_area = ((y1 + y2) / 2.0) * (x2 - x1)
        actual_area += trapezoid_area

    return actual_area

def calculate_aubc(data_points):
    """
    Calculates the Area Under the Budget-Accuracy Curve (AUBC).

    The AUBC is the area under the curve formed by (training_set_size, accuracy) points,
    normalized by the area of a perfect classifier (accuracy=1.0 over the same budget).
    :param data_points: A list of (size, accuracy) tuples.
    :return: The calculated AUBC value, or 0.0 if not calculable.
    """

    if len(data_points) < 2:
        # not enough points to calculate an area
        return 0.0

    actual_area = trapezoid_area(data_points)

    # the "perfect" area is a rectangle with height 1.0
    start_size = data_points[0][0]
    end_size = data_points[-1][0]
    total_budget_span = end_size - start_size

    # avoiding division by zero
    if total_budget_span == 0:
        return 0.0

    # normalized AUBC
    normalized_aubc = actual_area / float(total_budget_span)

    return normalized_aubc


def extract_metadata(filename):
    """
    Extracts dataset, method, and seed from the filename.
    :param filename: The filename string.
    :return: A tuple (dataset, method, seed), or (None, None, None) if malformed.
    """

    # leaving only the base part of the filename and splitting by '_'
    base_filename = filename.replace('_res.txt', '')
    parts = base_filename.split('_')

    # making sure we have enough parts to extract dataset, method, and seed
    if len(parts) < RESULTS_FILE_PART_NUM:
        print(f"Warning: Skipping malformed filename: {filename}")
        return None, None, None

    # extracting dataset, method, and seed
    return parts[DATASET_INDEX], parts[METHOD_INDEX], parts[SEED_INDEX]

def get_data_points(file_path, filename):
    """
    Extracts (training_set_size, accuracy) data points from the result file.
    :param file_path: The path to the result file.
    :param filename: The filename (for logging purposes).
    :return: A list of (size, accuracy) tuples, or None if no valid data found.
    """

    data_points = []
    # Regex to find lines like "Size of training set is 500, accuracy is 0.9321."
    line_pattern = re.compile(r"Size of training set is (\d+), accuracy is (\d+\.?\d*)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = line_pattern.search(line)
                if match:
                    size = int(match.group(1))
                    accuracy = float(match.group(2))
                    data_points.append((size, accuracy))

    except Exception as e:
        print(f"Warning: Could not read or parse file {filename}. Error: {e}")
        return None

    if not data_points:
        print(f"Warning: No accuracy data found in {filename}. Skipping.")
        return None

    return data_points

def process_file(file_path):
    """
    Parses a single result file to extract metadata and calculate metrics.
    :param file_path: The path to the res.txt file.
    :return: A dictionary containing the dataset, method, AUBC, F-acc,
              and the raw data points, or None if the file is invalid.
    """

    filename = os.path.basename(file_path)

    dataset, al_method, seed = extract_metadata(filename)

    # skipping files with malformed names
    if dataset is None:
        return None

    data_points = get_data_points(file_path, filename)

    # skipping files with no valid data points or wrong format
    if data_points is None:
        return None

    # calculating metrics from the data points
    final_accuracy = data_points[-1][1]
    aubc = calculate_aubc(data_points)

    return {
        'dataset': dataset,
        'method': al_method,
        'AUBC': aubc,
        'F-acc': final_accuracy,
        'seed': seed,
        'data_points': data_points
    }


def generate_plots(df):
    """
    Generates and saves learning curve plots for each dataset.
    :param df: A DataFrame containing the parsed results,including a 'data_points' column.
    """

    print("\n--- Generating Learning Curve Plots ---")

    # exploding the DataFrame to have one row per (size, accuracy) point
    plot_df = df.explode('data_points')
    plot_df[['size', 'accuracy']] = pd.DataFrame(plot_df['data_points'].tolist(), index=plot_df.index)
    plot_df = plot_df.astype({'size': int, 'accuracy': float})

    # averaging the accuracies across seeds for each (dataset, method, size)
    agg_df = plot_df.groupby(['dataset', 'method', 'size'])['accuracy'].mean().reset_index()

    # getting a list of unique datasets
    datasets = agg_df['dataset'].unique()

    for dataset_name in datasets:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # filtering data for the current dataset
        dataset_df = agg_df[agg_df['dataset'] == dataset_name]

        # plotting a line for each AL method
        for method_name in sorted(dataset_df['method'].unique()):
            method_df = dataset_df[dataset_df['method'] == method_name].sort_values('size')
            ax.plot(method_df['size'], method_df['accuracy'], marker='o', linestyle='-', label=method_name)

        # formatting the plot
        ax.set_title(f'Active Learning Performance on {dataset_name} Dataset', fontsize=16)
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(title='AL Method')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # saving the plot
        plot_filename = f"{dataset_name}_learning_curve.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"Saved plot: {plot_filename}")

    print("--- Finished Plot Generation ---\n")


def parse_args():
    """
    Parses command-line arguments to get the target directory.
    :return: The target directory path (if valid), or None if invalid.
    """

    parser = argparse.ArgumentParser(
        description="Analyze active learning results and generate a summary table and plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "directory",
        nargs='?',
        default='.',
        help="The directory containing the '*res.txt' files."
    )

    args = parser.parse_args()

    target_dir = args.directory

    # validating the directory
    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        return None

    return target_dir

def get_results_files(target_dir):
    """
    Finds and processes all '*res.txt' files in the specified directory.
    :param target_dir: The directory to search for result files.
    :return: A DataFrame containing the parsed results, or None if no valid files found.
    """

    # finding all relevant files
    result_files = [
        os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('res.txt')
    ]

    if not result_files:
        print("No '*res.txt' files found in the specified directory.")
        return None

    # processing all files and collecting the results
    all_results = []
    for f in result_files:
        result = process_file(f)
        if result:
            all_results.append(result)

    if not all_results:
        print("Could not extract any valid data from the found files.")
        return None

    df = pd.DataFrame(all_results)

    return df


def create_summary_table(df):
    """
    Creates and prints a summary table of AUBC and final accuracy for each (dataset, method) pair.
    :param df: A DataFrame containing the parsed results.
    """

    # for the summary table, we don't need the detailed data points anymore
    summary_df = df.drop(columns=['data_points'])

    # averaging over seeds if multiple seeds exist for the same (dataset, method)
    summary_df = summary_df.groupby(['dataset', 'method']).agg({'AUBC': 'mean', 'F-acc': 'mean'}).reset_index()

    # pivoting the table to get the desired structure:
    #   Rows: AL methods
    #   Columns: Datasets with sub-columns for AUBC and F-acc for each dataset
    pivot_df = summary_df.pivot_table(
        index='method',
        columns='dataset',
        values=['AUBC', 'F-acc']
    )

    # reordering columns to group by dataset
    if not pivot_df.empty:
        pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    print("--- Active Learning Results Summary ---")
    # formatting the float values for better readability
    pd.options.display.float_format = '{:.4f}'.format
    print(pivot_df.to_string())
    print("\n--- End of Summary ---")


def main():
    """
    Main function to find files, process them, and print the summary table.
    """

    # parsing command-line arguments
    target_dir = parse_args()

    # target directory is invalid
    if target_dir is None:
        return

    # extracting results from files
    df = get_results_files(target_dir)

    # no valid data extracted
    if df is None:
        return

    # generating plots
    generate_plots(df.copy())

    # creating and printing the summary table
    create_summary_table(df)

if __name__ == "__main__":
    main()