import os
import argparse
import enum
import json
import string
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

class View(enum.Enum):
    Overhead = 'overhead_view'
    Vehicle = 'vehicle_view'

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="./data/caption/train",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis/output",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--view",
        type=View,
        default=View.Overhead,
        choices=list(View),
        help="Point of view of video to analyze",
    )

    return parser.parse_args()


def run_analysis(output_dir, caption_dir, view):
    print("Start processing...")
    fps = 10
    histogram_data = {}

    for vid in tqdm(os.listdir(caption_dir)):
        caption_path = os.path.join(caption_dir, vid, view)
        try:
            for caption_file in os.listdir(caption_path):
                file_path = os.path.join(caption_path, caption_file)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    events = data["event_phase"]

                    for event in events:
                        start_time = float(event["start_time"])
                        end_time = float(event["end_time"])
                        label = str(event['labels'][0])
                        start = round(start_time * fps)
                        end = round(end_time * fps)
                        histogram_data.setdefault(label, []).append(end - start)

        except FileNotFoundError as e:
            # print(e)
            continue

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, value in histogram_data.items():
        output_path = os.path.join(output_dir, f"frames_analysis_label_{key}.png")
        fig, ax = plt.subplots()
        counts, bins = np.histogram(value)
        ax.hist(bins[:-1], bins, weights=counts)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('# Frames')

        plt.savefig(output_path)
        plt.clf()

    print("analysis successfully")


if __name__ == '__main__':
    """
        python ./analysis/frame_analysis.py
    """
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    CAPTION_DIR = args.caption_dir
    VIEW = str(args.view)

    run_analysis(OUTPUT_DIR, CAPTION_DIR, VIEW)
