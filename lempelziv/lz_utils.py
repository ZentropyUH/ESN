import json
import os
import random
import string
import subprocess
# import tempfile
from itertools import product
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import List

import pandas as pd


import time

############# HELPER FUNCTIONS #############

def _generate_random_filename(directory, suffix='', length=10):
    """
    Generate a random filename in the specified directory.

    Args:
        directory (str): The directory to create the file in.
        suffix (str, optional): Suffix for the filename.
        length (int, optional): Length of the random part of the filename.

    Returns:
        str: The generated random filename.
    """
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    filename = os.path.join(directory, f"{random_part}{suffix}")
    return filename

def _convert_to_numeric(value):
    """
    Convert a string to int or float, if possible.

    Args:
        value (str): The string to convert.

    Returns:
        int or float or str: The converted value, or the original string if conversion fails.
    """
    try:
        # First, try to convert to int
        return int(value)
    except ValueError:
        try:
            # If int conversion fails, try to convert to float
            return float(value)
        except ValueError:
            # If both conversions fail, return the original string
            return value

def _csv_to_bin_df(filepath: str, method: str ='mean')-> pd.DataFrame:
    """
    Binarize the columns of a CSV file based on the mean or median.

    Args:
        filepath (str): Path to the CSV file.
        method (str): Method for binarization ('mean' or 'median').

    Returns:
        pd.DataFrame: Binarized DataFrame.
    """
    # Read the CSV file
    df = pd.read_csv(filepath, header=None)

    # Check if method is valid
    if method not in ['mean', 'median']:
        raise ValueError("Method should be 'mean' or 'median'")

    # Binarize based on the specified method
    if method == 'mean':
        threshold = df.mean()
    elif method == 'median':
        threshold = df.median()
    elif isinstance(method, float):
        threshold = method

    # Apply binarization
    binarized_df = (df > threshold).astype(int)

    return binarized_df

def _df_to_strings(df: pd.DataFrame, max_parallel_tasks: int = 16) -> List[str]:
    """
    Convert binary DataFrame columns to a list of binary sequences in parallel.

    Args:
        df (pd.DataFrame): DataFrame with binary values (0s and 1s).
        max_parallel_tasks (int): Number of processes to use. Default is None, which uses os.cpu_count().

    Returns:
        list: List of binary sequences as strings.
    """
    tasks = [df[column] for column in df.columns]
    with ThreadPool(processes=max_parallel_tasks) as pool:
        sequences = pool.map(_worker_bin_column2str, tasks)
    return sequences

def _process_lz76(text: str) -> dict:
    """
    Parse the output of lempelziv program and return a dictionary with the parsed data.

    The values in the dictionary are lists of numeric values.

    Args:
        text (str): The output of lempelziv program.

    Returns:
        dict: Dictionary with the parsed data.
    """
    parsed_data = {}
    lines = text.split('\n')
    for line in lines:
        parts = line.split(':')
        if len(parts) == 2:
            key = parts[0].strip().strip('[]').strip()
            # Splitting the values part into a list of values
            values = parts[1].strip().strip('[]').strip().split()
            # Convert each value in the list to numeric if possible
            parsed_data[key] = [_convert_to_numeric(value) for value in values]
    return parsed_data

def _process_multiple_binary_sequences_lz(binary_sequences: List[str], flags: List[str] = None, max_parallel_tasks=16) -> dict:
    """
    Process a list of binary strings using lempelziv with optional flags,
    using a randomly generated temporary filename.

    Args:
        binary_sequences (list): List of binary strings to process.
        flags (list, optional): List of flags to pass to the binary program.

    Returns:
        dict: List of dictionaries with the parsed data.
    """
    parsed_data = {}

    n = len(binary_sequences)

    tasks = [(binary_sequences, idx, flags) for idx in range(n)]

    with ThreadPool(processes=max_parallel_tasks) as pool:
        results = pool.map(_worker_single_sequence, tasks)

    for idx, lz_data in results:
        parsed_data[idx] = lz_data

    return parsed_data

# This does the LZ processing for two binary sequences compared
def _lz_distance(sequence1: str, sequence2: str) -> float:
    """
    Process two binary sequences using lempelziv with optional flags.

    Args:
        sequence1 (str): The first binary sequence to process.
        sequence2 (str): The second binary sequence to process.
        flags (list, optional): List of flags to pass to the binary program.

    Returns:
        float: Dictionary with the parsed data.
    """
    
    # Generate a random input filename in the current directory
    input_filename = _generate_random_filename('.', '.txt')

    # Write the binary strings to the file
    with open(input_filename, 'w', encoding='utf-8') as file:
        file.write(sequence1 + '\n' + sequence2)


    # Prepare and run the command
    command = ['lempelziv', '-d', '-u', input_filename]
    subprocess.run(command, check=True)

    output_filename = input_filename + '.lz76'
    
    # Read and delete the output file
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as file:
            output_content = file.read()
        os.remove(output_filename)
    else:
        raise FileNotFoundError(f"Output file {output_filename} not found")

    # Delete the input file
    os.remove(input_filename)
    
    parsed_data = _process_lz76(output_content)
    
    return parsed_data["Normalized LZ76 distance between succesive lines"][0]

############# WORKERS FOR PARALLELIZATION #############

# This does the LZ processing for a single binary sequence
def _worker_single_sequence(args):

    binary_sequences, idx, flags = args

    # Generate a random input filename in the current directory
    input_filename = _generate_random_filename('.', '.txt')

    # Write the binary string to the file
    with open(input_filename, 'w', encoding='utf-8') as file:
        file.write(binary_sequences[idx])

    output_filename = input_filename + '.lz76'

    # Prepare and run the command
    command = ['lempelziv']
    if flags:
        command.extend(flags)
    command.append(input_filename)
    subprocess.run(command, check=True)

    # Read and delete the output file
    output_content = ''
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as file:
            output_content = file.read()
        os.remove(output_filename)
    else:
        raise FileNotFoundError(f"Output file {output_filename} not found")

    # Delete the input file
    os.remove(input_filename)

    parsed_output = _process_lz76(output_content)

    return idx, parsed_output

# This does the LZ processing for two binary sequences compared
def _worker_distance_2_sequences(args):
    
    idx, sequence1, sequence2 = args
    
    distance = _lz_distance(sequence1, sequence2)
    
    return idx, distance

def _worker_single_file(args):
    """
    Worker task to process a single CSV file using lempelziv.
    """
    csv_file, method, flags = args

    return os.path.basename(csv_file), lz_csv(csv_file, method, flags)

def _worker_distance_between_csvs(args):
    """
    Worker task to calculate distance between two files.
    """
    file1, file2, method = args
    distances = lz_distance_csv(file1, file2, method)
    return file1, file2, distances

def _worker_distance_single_csv(args):
    """
    Worker task to calculate distance between two columns.
    """
    binary_sequences, idx1, idx2 = args
    return (idx1, idx2, _lz_distance(binary_sequences[idx1], binary_sequences[idx2]))

def _worker_bin_column2str(column_data: pd.Series) -> str:
    """
    Worker function to process a single column and convert it to a binary sequence.

    Args:
        column_data (pd.Series): Series with binary values (0s and 1s).

    Returns:
        str: Binary sequence as a string.
    """
    return ''.join(column_data.astype(str))

############# FUNCTIONS TO BE USED #############

def binary_sequences_from_csv(filepath: str, method: str ='mean', save_path: str = None, max_parallel_tasks: int = 16) -> List[str]:
    """
    Convert each column of a binary CSV file to a binary string.

    Args:
        filepath (str): Path to the CSV file.
        method (str): Method for binarization ('mean' or 'median').
        save_path (str, optional): Path to the folder to save the binary sequences.

    Returns:
        list: List of binary strings.
    """
    # Binarize the CSV file
    binarized_df = _csv_to_bin_df(filepath, method)

    # Convert to binary sequences
    binary_sequences = _df_to_strings(binarized_df, max_parallel_tasks=max_parallel_tasks)

    if save_path is not None:
        print(f"Saving to {save_path}.")
        # Save the binary sequences to files
        with open(save_path, 'w', encoding='utf-8') as file:
            for line in binary_sequences:
                file.write(line + '\n')

    return binary_sequences

def lz_csv(file_path: str, method='mean', flags=None, save_path: str = None, max_parallel_tasks: int = 16) -> dict:
    """
    Process a CSV file using lempelziv with optional flags.

    Args:
        file_path (str): Path to the CSV file.
        method (str): Method for binarization ('mean' or 'median').
        flags (list, optional): List of flags to pass to the binary program.
        save_path (str, optional): Path to the json file to save the results.

    Returns:
        dict: List of dictionaries with the parsed data.
    """
    # Binarize the CSV file
    binarized_df = _csv_to_bin_df(file_path, method)

    # Convert to binary sequences
    binary_sequences = _df_to_strings(binarized_df)

    # Process the binary sequences
    results = _process_multiple_binary_sequences_lz(binary_sequences, flags, max_parallel_tasks=max_parallel_tasks)

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4)

    return results

def lz_distance_csv(csv_file1: str, csv_file2: str, method: str = 'mean', max_parallel_tasks: int = 16) -> dict:
    """
    Calculate the LZ distance for each column between two CSV files.

    Args:
        csv_file1 (str): Path to the first CSV file.
        csv_file2 (str): Path to the second CSV file.
        method (str): Method for binarization ('mean' or 'median').

    Returns:
        dict: A dictionary where keys are column indices and values are LZ distances.
    """
    # Binarize both CSV files
    binarized_csv_1 = binary_sequences_from_csv(csv_file1, method)
    binarized_csv_2 = binary_sequences_from_csv(csv_file2, method)

    assert len(binarized_csv_1) == len(binarized_csv_2), "CSV files must have the same number of columns"

    # Calculate LZ distance for each pair of corresponding columns
    lz_distances = {}
    
    tasks = [(idx, sequence1, sequence2) for idx, (sequence1, sequence2) in enumerate(zip(binarized_csv_1, binarized_csv_2))]
    
    with ThreadPool(processes=max_parallel_tasks) as pool:
        results = pool.map(_worker_distance_2_sequences, tasks)
    
    for idx, distance in results:
        lz_distances[idx] = distance
    
    return lz_distances

def lz_folder(folder_path: str, method: str = 'mean', flags=None, save_path: str = None, max_parallel_tasks: int = 16) -> dict:
    """
    Parallelized version of processing all CSV files in a folder using lempelziv with optional flags.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        method (str): Method for binarization ('mean' or 'median').
        flags (list, optional): List of flags to pass to the binary program.
        save_path (str, optional): Path to the json file to save the results.
        max_parallel_tasks (int): The maximum number of parallel tasks.

    Returns:
        dict: A dictionary where keys are file names and values are LZ processing results.
    """
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]

    # Prepare the tasks for parallel processing
    tasks = [(csv_file, method, flags) for csv_file in csv_paths]

    # Process the files in parallel
    with Pool(processes=max_parallel_tasks) as pool:
        results_list = pool.map(_worker_single_file, tasks)

    # Organize the results into a dictionary
    results = {file_name: result for file_name, result in results_list}

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4)

    return results

def distance_matrices_between_folders(folder_path1: str, folder_path2: str=None, method='mean', save_path: str=None, max_parallel_tasks: int = 16) -> str:
    """
    Parallelized version of calculating distance matrices between all CSV files in two folders.
    """
    # List all CSV files in both folders
    csv_files_folder1 = [file for file in os.listdir(folder_path1) if file.endswith('.csv')]
    csv_paths_folder1 = [os.path.join(folder_path1, file) for file in csv_files_folder1]
    
    if folder_path2 is not None:
        csv_files_folder2 = [file for file in os.listdir(folder_path2) if file.endswith('.csv')]
        csv_paths_folder2 = [os.path.join(folder_path2, file) for file in csv_files_folder2]

    # Generate all pairs of files, one from each folder
    if folder_path2 is not None:
        file_pairs = list(product(csv_paths_folder1, csv_paths_folder2))
    else: 
        file_pairs = list(product(csv_paths_folder1, repeat=2))

    # Prepare tasks for parallel processing
    tasks = [(file1, file2, method) for file1, file2 in file_pairs]

    # Initialize a dictionary to hold the distance matrices
    distance_matrices = {}

    with Pool(processes=max_parallel_tasks) as pool:
        results = pool.map(_worker_distance_between_csvs, tasks)

    # Process results
    for file1, file2, distances in results:
        print(f"Distances between {file1} and {file2}: {distances}")

        # Aggregate distances into the distance_matrices dictionary
        for col_idx, distance in distances.items():
            if col_idx not in distance_matrices:
                distance_matrices[col_idx] = pd.DataFrame(
                    index=[os.path.basename(file) for file in csv_files_folder1],
                    columns=[os.path.basename(file) for file in csv_files_folder2] if folder_path2 is not None else [os.path.basename(file) for file in csv_files_folder1]
                )

            distance_matrices[col_idx].loc[os.path.basename(file1), os.path.basename(file2)] = distance

    # Convert the distance matrices to JSON
    distance_matrices_dict = {col_idx: df.to_dict() for col_idx, df in distance_matrices.items()}

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(distance_matrices_dict, file, indent=4)
                
    return distance_matrices_dict

def distance_columns_single_csv(csv_file: str, method='mean', save_path: str=None, max_parallel_tasks: int = 16) -> str:
    """
    Parallelized version of calculating distance matrices per column.
    """
    # Binarize the CSV file
    print(f"Binarizing CSV file: {csv_file}\n")
    binarized_df = _csv_to_bin_df(csv_file, method)

    # Convert to binary sequences
    print("Converting to List of string binary sequences.\n")
    binary_sequences = _df_to_strings(binarized_df)
    print(f"Binarized. Number of columns: {len(binary_sequences)}\n")

    # Initialize an pd.Dataframe to hold the distance matrices
    n = len(binary_sequences)
    distance_matrix = pd.DataFrame(index=range(n), columns=range(n))

    # Prepare the list of tasks
    tasks = [(binary_sequences, idx1, idx2) for idx1 in range(n) for idx2 in range(n)]

    # Using multiprocessing pool to parallelize the task
    with ThreadPool(processes=max_parallel_tasks) as pool:
        print(f"Calculating distances between all pairs of columns in {csv_file}.")
        results = pool.map(_worker_distance_single_csv, tasks)

    # Fill the distance matrix with results
    for idx1, idx2, distance in results:
        distance_matrix.loc[idx1, idx2] = distance
        print(f"Distance calculated between columns {idx1} and {idx2}.")

    if save_path is not None:
        # save the results to a csv file
        distance_matrix.to_csv(save_path, index=False)
            
    return distance_matrix