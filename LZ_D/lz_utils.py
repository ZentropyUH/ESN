import pandas as pd
from typing import List
import subprocess
import os
import tempfile
from itertools import product
import json

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
        
def _binarize_csv(filepath: str, method: str ='mean')-> pd.DataFrame:
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
    else:
        threshold = df.median()

    # Apply binarization
    binarized_df = (df > threshold).astype(int)

    return binarized_df

def _binary_sequences_from_df(df: pd.DataFrame) -> List[str]:
    """
    Convert binary DataFrame columns to a list of binary sequences.

    Args:
        df (pd.DataFrame): DataFrame with binary values (0s and 1s).

    Returns:
        list: List of binary sequences as strings.
    """
    sequences = []
    for column in df.columns:
        sequence = ''.join(df[column].astype(str))
        sequences.append(sequence)
    return sequences

# This does the LZ processing for a single binary sequence
def _process_single_binary_sequence_lz(binary_string, flags=None):
    """
    Process a binary string using lempelziv with optional flags,
    using a randomly generated temporary filename.

    Args:
        binary_string (str): The binary string to process.
        flags (list, optional): List of flags to pass to the binary program.

    Returns:
        str: The content of the generated file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='') as temp_file:
        input_filename = temp_file.name
        temp_file.write(binary_string)

    output_filename = input_filename + '.lz76'

    # Prepare the command for subprocess
    command = ['lempelziv']
    if flags:
        command.extend(flags)
    command.append(input_filename)

    # Call the external binary program
    subprocess.run(command, check=True)

    # Read the output from the generated file
    output_content = ''
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as file:
            output_content = file.read()

        # Clean up the output file
        os.remove(output_filename)
    
    else:
        raise FileNotFoundError(f"Output file {output_filename} not found")

    # Clean up the input temporary file
    os.remove(input_filename)

    return output_content

# This does the LZ processing for two binary sequences compared
def _process_dual_binary_sequence_lz(string1, string2):
    """
    Process two binary strings using lempelziv with specific flags,
    using a randomly generated temporary filename. The strings are
    written as consecutive lines in the file.

    Args:
        string1 (str): The first binary string to process.
        string2 (str): The second binary string to process.

    Returns:
        str: The content of the generated file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='') as temp_file:
        input_filename = temp_file.name
        temp_file.write(string1 + '\n' + string2)

    output_filename = input_filename + '.lz76'

    # Prepare the command for subprocess
    command = ['lempelziv', '-d', '-u', input_filename]

    # Call the external binary program
    subprocess.run(command, check=True)

    # Read the output from the generated file
    output_content = ''
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as file:
            output_content = file.read()

        # Clean up the output file
        os.remove(output_filename)
    else:
        raise FileNotFoundError(f"Output file {output_filename} not found")

    # Clean up the input temporary file
    os.remove(input_filename)

    return output_content

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

def _process_multiple_binary_sequences_lz(binary_sequences: List[str], flags: List[str] = None) -> dict:
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
    for idx, binary_sequence in enumerate(binary_sequences):
        lz76_output = _process_single_binary_sequence_lz(binary_sequence, flags)
        parsed_data[idx] = _process_lz76(lz76_output)
    return parsed_data

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
    lz76_output = _process_dual_binary_sequence_lz(sequence1, sequence2)
    parsed_data = _process_lz76(lz76_output)
    return parsed_data["Normalized LZ76 distance between succesive lines"]

def binary_sequences_from_csv(filepath: str, method: str ='mean', save_path: str = None) -> List[str]:
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
    binarized_df = _binarize_csv(filepath, method)

    # Convert to binary sequences
    binary_sequences = _binary_sequences_from_df(binarized_df)
    
    if save_path is not None:
        # Save the binary sequences to files
        for idx, sequence in enumerate(binary_sequences):
            with open(os.path.join(save_path, f"{idx}.txt"), "w", encoding='utf-8') as file:
                file.write(sequence)
    
    return binary_sequences

def lz_csv(file_path, method='mean', flags=None, save_path: str = None) -> dict:
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
    binarized_df = _binarize_csv(file_path, method)

    # Convert to binary sequences
    binary_sequences = _binary_sequences_from_df(binarized_df)

    # Process the binary sequences
    results = _process_multiple_binary_sequences_lz(binary_sequences, flags)
    
    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(results, file)
    
    return results

def lz_distance_csv(csv_file1: str, csv_file2: str, method='mean') -> dict:
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
    binary_sequences1 = binary_sequences_from_csv(csv_file1, method)
    binary_sequences2 = binary_sequences_from_csv(csv_file2, method)


    n = len(binary_sequences1)

    # Ensure both CSV files have the same number of columns
    if len(binary_sequences1) != len(binary_sequences2):
        raise ValueError("CSV files must have the same number of columns")

    # Calculate LZ distance for each pair of corresponding columns
    lz_distances = {}
    for idx in range(n):
        sequence1 = binary_sequences1[idx]
        sequence2 = binary_sequences2[idx]
        lz_distances[idx] = _lz_distance(sequence1, sequence2)

    return lz_distances

def lz_folder(folder_path: str, method='mean', flags=None, save_path: str = None) -> dict:
    """
    Process all CSV files in a folder using lempelziv with optional flags.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        method (str): Method for binarization ('mean' or 'median').
        flags (list, optional): List of flags to pass to the binary program.
        save_path (str, optional): Path to the json file to save the results.

    Returns:
        dict: List of dictionaries with the parsed data.
    """
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]

    # Initialize a dictionary to hold the results
    results = {}

    for csv_file in csv_paths:
        print(f"Processing CSV file: {csv_file}\n")
        results[os.path.basename(csv_file)] = lz_csv(csv_file, method, flags)

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(results, file)

    return results

def distance_matrices_single_folder(folder_path: str, method='mean', save_path: str=None) -> str:
    """
    Calculate distance matrices per column between all CSV files in a folder.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        method (str): Method for binarization ('mean' or 'median').
        save_path (str, optional): Path to the json file to save the results.

    Returns:
        str: JSON string representing the distance matrices.
    """
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]

    # Generate all pairs of files
    file_pairs = list(product(csv_paths, repeat=2))

    # Initialize a dictionary to hold the distance matrices
    distance_matrices = {}

    for file1, file2 in file_pairs:
        distances = lz_distance_csv(file1, file2, method)
        
        print(f"Distances between {file1} and {file2}: {distances}")

        # Aggregate distances into the distance_matrices dictionary
        for col_idx, distance in distances.items():
            if col_idx not in distance_matrices:
                distance_matrices[col_idx] = pd.DataFrame(index=csv_files, columns=csv_files)
            distance_matrices[col_idx].loc[os.path.basename(file1), os.path.basename(file2)] = distance

    # Convert the distance matrices to JSON
    distance_matrices_json = {col_idx: df.to_json() for col_idx, df in distance_matrices.items()}

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(distance_matrices_json, file)

    return distance_matrices_json

def distance_matrices_between_folders(folder1_path: str, folder2_path: str, method='mean', save_path: str=None) -> str:
    """
    Calculate distance matrices per column between all CSV files in two folders.

    Args:
        folder1_path (str): Path to the first folder containing CSV files.
        folder2_path (str): Path to the second folder containing CSV files.
        method (str): Method for binarization ('mean' or 'median').

    Returns:
        str: JSON string representing the distance matrices.
    """
    # List all CSV files in both folders
    csv_files_folder1 = [file for file in os.listdir(folder1_path) if file.endswith('.csv')]
    csv_files_folder2 = [file for file in os.listdir(folder2_path) if file.endswith('.csv')]
    csv_paths_folder1 = [os.path.join(folder1_path, file) for file in csv_files_folder1]
    csv_paths_folder2 = [os.path.join(folder2_path, file) for file in csv_files_folder2]

    # Generate all pairs of files, one from each folder
    file_pairs = list(product(csv_paths_folder1, csv_paths_folder2))

    # Initialize a dictionary to hold the distance matrices
    distance_matrices = {}

    for file1, file2 in file_pairs:
        distances = lz_distance_csv(file1, file2, method)

        print(f"Distances between {file1} and {file2}: {distances}")
        
        # Aggregate distances into the distance_matrices dictionary
        for col_idx, distance in distances.items():
            
            if col_idx not in distance_matrices:
                distance_matrices[col_idx] = pd.DataFrame(
                    index=csv_files_folder1,
                    columns=csv_files_folder2
                    )
            
            distance_matrices[col_idx].loc[os.path.basename(file1), os.path.basename(file2)] = distance

    # Convert the distance matrices to JSON
    distance_matrices_json = {col_idx: df.to_json() for col_idx, df in distance_matrices.items()}

    if save_path is not None:
        # Save the results to a json file
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(distance_matrices_json, file)

    return distance_matrices_json
