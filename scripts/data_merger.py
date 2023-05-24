import os
import argparse
import pandas as pd

def process_csv(root_folder, output_file):
    all_data = []

    for subdir, dirs, files in os.walk(root_folder):
        csv_files = [f for f in files if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_file_path = os.path.join(subdir, csv_file)
            data = pd.read_csv(csv_file_path)
            all_data.append(data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        avg_data = combined_data.groupby('step').mean().reset_index()
        avg_data.to_csv(os.path.join(root_folder, output_file), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV files and create an averaged CSV file in the root folder')
    parser.add_argument('root_folder', type=str, help='Path to the root folder containing the CSV files')
    args = parser.parse_args()

    output_file = 'final_avg_data.csv'
    process_csv(args.root_folder, output_file)
