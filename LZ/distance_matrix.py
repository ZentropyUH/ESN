import sys
import pandas as pd
import os
import subprocess
import shutil
import json

from LZ.lz import lzProc

def create_distance_matrix(files_dir, output_dir, key="DLZ compare"):

    txt_files = [file for file in os.listdir(files_dir)]
    txt_files.sort(key=lambda x: int(x.split('.')[0]))
    
    # Initialize an empty matrix with 0s
    num_files = len(txt_files)
    distance_matrix = [[0] * num_files for _ in range(num_files)]
    
    # Fill the matrix with distances
    for i, primary_file in enumerate(txt_files):
        for j, secondary_file in enumerate(txt_files):
            if i < j:  # Avoid redundant computations
                primary_file_path = os.path.join(files_dir, primary_file)
                secondary_file_path = os.path.join(files_dir, secondary_file)

                # Return path to json with distance
                json_path = lzProc(output_dir, ("z", primary_file_path, secondary_file_path))
                
                # Loas json
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                distance = json_data.get(key, -1)

                # index in distance
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance  # Symmetric matrix

    # Convert the matrix to a DataFrame and save to CSV
    indexes = [file.replace(".txt", "") for file in txt_files]
     
    df = pd.DataFrame(distance_matrix, index=indexes, columns=indexes)
    df.to_csv(os.path.join(output_dir, 'distance_matrix.csv'))

if __name__ == "__main__":
    create_distance_matrix(sys.argv[1], sys.argv[2])
