import os
import pandas as pd
import sys

# Binarize a folder with csv files
def mean_per_col_for_folder(root_folder_path: str, save_folder_path: str):
    """
    Each csv file will be a folder in save folder path.
    Each column of the csv file will be a txt file inside that folder.
    """
    csv_files = [f for f in os.listdir(root_folder_path) if f.endswith('.csv')]
    print(f"CSV files found: {csv_files}\n")

    for csv_file in csv_files:
        file_path = os.path.join(root_folder_path, csv_file)
        print(f"Processing CSV file: {csv_file}\n")
        mean_per_col(file_path, save_folder_path)


# Binarize a csv file
def mean_per_col(path: str, save: str):
    data = pd.read_csv(path)

    base_name = os.path.basename(path)
    folder_path = os.path.join(save, base_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, column in enumerate(data.columns, start=1):
        mean_value = data[column].mean()
        bin_col = [1 if num > mean_value else 0 for num in data[column]]

        bin_str = ''.join(map(str, bin_col))
        with open(os.path.join(folder_path, f"{i}.txt"), "w") as f:
             f.write(bin_str)


if __name__ == "__main__":
    mean_per_col_for_folder(sys.argv[1], sys.argv[2])

