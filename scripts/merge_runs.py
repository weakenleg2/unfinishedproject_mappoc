import os
import argparse
import pandas as pd

def process_csv(root_folder, output_file, metric):
    all_data = []

    for subdir, dirs, _ in os.walk(root_folder):
        metric_path = os.path.join(subdir, "logs", metric)
        if os.path.exists(metric_path):
            csv_file_path = os.path.join(metric_path, "final_avg_data.csv")
            if os.path.isfile(csv_file_path):
                data = pd.read_csv(csv_file_path)
                all_data.append(data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        avg_data = combined_data.groupby('step').mean().reset_index()
        avg_data.to_csv(os.path.join(root_folder, output_file), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV files and create an averaged CSV file in the root folder')
    parser.add_argument('root_folder', type=str, help='Path to the root folder containing the subfolders')
    parser.add_argument('metric', type=str, help='Metric to average')
    args = parser.parse_args()

    output_file = f'final_avg_{args.metric}.csv'
    process_csv(args.root_folder, output_file, args.metric)
