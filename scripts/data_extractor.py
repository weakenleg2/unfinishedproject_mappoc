import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def event_file_to_csv(event_file_path, csv_file_path):
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()

    data_frames = []
    for tag in event_acc.Tags()['scalars']:
        tag_data = [(event.step, event.value) for event in event_acc.Scalars(tag)]
        df = pd.DataFrame(tag_data, columns=['step', tag])
        data_frames.append(df)

    if data_frames:
        data = pd.concat(data_frames, axis=1)
        data.to_csv(csv_file_path, index=False)

def process_subfolders(root_folder):
    for subdir, dirs, files in os.walk(root_folder):
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        for event_file in event_files:
            event_file_path = os.path.join(subdir, event_file)
            csv_file_path = os.path.join(subdir, f"{os.path.splitext(event_file)[0]}.csv")
            event_file_to_csv(event_file_path, csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TensorBoard event files to CSV files')
    parser.add_argument('root_folder', type=str, help='Path to the root folder containing the event files')
    args = parser.parse_args()

    process_subfolders(args.root_folder)
