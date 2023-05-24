import os
import argparse
import pandas as pd
import sys

def process_csv(root_folder, output_file, metric):
    all_data = []

    for subdir, dirs, files in os.walk(root_folder):
        csv_files = [f for f in files if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_file_path = os.path.join(subdir, csv_file)
            data = pd.read_csv(csv_file_path)
            all_data.append(data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        if metric not in combined_data.columns:
            print("Nah, couldn't find metric")
            sys.exit(1)
            
        avg_data = combined_data.groupby('step')[['step', metric]].mean().reset_index()
        avg_data.to_csv(os.path.join(root_folder, output_file), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV files and create an averaged CSV file in the root folder')
    parser.add_argument('root_folder', type=str, help='Path to the root folder containing the CSV files')
    parser.add_argument('metric', type=str, help='Metric to average')
    args = parser.parse_args()

    output_file = f'final_avg_{args.metric}.csv'
    process_csv(args.root_folder, output_file, args.metric)
