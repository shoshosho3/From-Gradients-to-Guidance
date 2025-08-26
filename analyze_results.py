import os
import re
import argparse
import pandas as pd
from collections import defaultdict

def calculate_aubc(data_points):
    """
    Calculates the Area Under the Budget-Accuracy Curve (AUBC).

    The AUBC is the area under the curve formed by (training_set_size, accuracy) points,
    normalized by the area of a perfect classifier (accuracy=1.0 over the same budget).

    Args:
        data_points (list): A list of (size, accuracy) tuples.

    Returns:
        float: The calculated AUBC value, or 0.0 if not calculable.
    """
    if len(data_points) < 2:
        # Not enough points to calculate an area
        return 0.0

    # Sort points by training set size to be safe
    data_points.sort(key=lambda x: x[0])

    actual_area = 0.0
    # Use the trapezoidal rule to calculate the area under the curve
    for i in range(len(data_points) - 1):
        x1, y1 = data_points[i]
        x2, y2 = data_points[i+1]
        
        # Area of one trapezoid: (average height) * width
        trapezoid_area = ((y1 + y2) / 2.0) * (x2 - x1)
        actual_area += trapezoid_area

    # The "perfect" area is a rectangle with height 1.0
    start_size = data_points[0][0]
    end_size = data_points[-1][0]
    total_budget_span = end_size - start_size

    if total_budget_span == 0:
        return 0.0 # Avoid division by zero

    perfect_area = 1.0 * total_budget_span
    
    return actual_area / perfect_area

def process_file(file_path):
    """
    Parses a single result file to extract metadata and calculate metrics.

    Args:
        file_path (str): The path to the res.txt file.

    Returns:
        dict: A dictionary containing the dataset, method, AUBC, and F-acc,
              or None if the file is invalid.
    """
    filename = os.path.basename(file_path)
    
    # 1. Extract metadata from the filename
    # e.g., MNIST_LEGL_250_500_10000_normal_res_tot.txt
    parts = filename.split('_')
    if len(parts) < 2:
        print(f"Warning: Skipping malformed filename: {filename}")
        return None
        
    dataset = parts[0]
    al_method = parts[1]

    # 2. Parse file content for accuracy data
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
        
    # 3. Calculate metrics
    # Sort by size to ensure final accuracy is from the largest set size
    data_points.sort(key=lambda x: x[0])
    
    final_accuracy = data_points[-1][1]
    aubc = calculate_aubc(data_points)

    return {
        'dataset': dataset,
        'method': al_method,
        'AUBC': aubc,
        'F-acc': final_accuracy
    }

def main():
    """
    Main function to find files, process them, and print the summary table.
    """
    parser = argparse.ArgumentParser(
        description="Analyze active learning results and generate a summary table.",
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
    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        return

    print(f"Searching for '*res.txt' files in: {target_dir}\n")

    # Find all relevant files
    result_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('res.txt')]

    if not result_files:
        print("No '*res.txt' files found in the specified directory.")
        return

    # Process all files and collect the results
    all_results = []
    for f in result_files:
        result = process_file(f)
        if result:
            all_results.append(result)

    if not all_results:
        print("Could not extract any valid data from the found files.")
        return

    # Use pandas to create and format the summary table
    df = pd.DataFrame(all_results)
    
    # Pivot the table to get the desired structure:
    # Rows: AL methods
    # Columns: Datasets with sub-columns for AUBC and F-acc
    pivot_df = df.pivot_table(
        index='method', 
        columns='dataset', 
        values=['AUBC', 'F-acc']
    )

    # Reorder columns to group by dataset (e.g., MNIST AUBC, MNIST F-acc, ...)
    pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    print("--- Active Learning Results Summary ---")
    # Format the float values for better readability
    pd.options.display.float_format = '{:.4f}'.format
    print(pivot_df.to_string())
    print("\n--- End of Summary ---")


if __name__ == "__main__":
    main()