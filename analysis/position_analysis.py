import os
import argparse
import enum
import json
import string
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer


class View(enum.Enum):
    Overhead = 'overhead_view'
    Vehicle = 'vehicle_view'

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="./datasets/output",
        help="Path to the directory containing Loc's analysis",
    )

    parser.add_argument(
        "--caption_dir",
        type=str,
        default="./datasets/wts_dataset_zip/annotations/caption",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--view",
        type=View,
        default=View.Overhead,
        choices=list(View),
        help="Point of view of video to analyze",
    )

    parser.add_argument(
        "--bbox_dir",
        type=str,
        default="./datasets/wts_dataset_zip/annotations/bbox_annotated",
        help="Path to the directory containing WTS bbox",
    )

    parser.add_argument(
        "--ngram",
        type=int,
        default=1,
        help="ngram to analysis word frequency",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/output",
        help="Path to the directory containing position analysis",
    )

    parser.add_argument("--stopword",
                        action="store_true")

    return parser.parse_args()


def run_analysis(analysis_dir, output_dir, caption_dir, bbox_dir, view, ngram):
    print("Start processing...")

    phases = ["train", "val"]
    keyword = ["left", "right", "front", "diagonally"]
    position_analysis = {
        "train": [],
        "val": []
    }

    for phase in phases:

        current_bbox_dir = os.path.join(bbox_dir, "pedestrian", phase)
        current_caption_dir = os.path.join(caption_dir, phase)
        current_output_dir = os.path.join(output_dir, phase)

        for vid in tqdm(os.listdir(current_bbox_dir)):
            bbox_vehical_path = os.path.join(bbox_dir, "vehicle", phase, vid, view)
            bbox_pedestrian_path = os.path.join(bbox_dir, "pedestrian", phase, vid, view)
            caption_path = os.path.join(current_caption_dir, vid, view)
            output_path = os.path.join(current_output_dir, vid)
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            try:
                for caption_file in os.listdir(caption_path):
                    caption_file_path = os.path.join(caption_path, caption_file)

                    bbox_pedestrian_lst = os.listdir(bbox_pedestrian_path)
                    bbox_vehical_lst = os.listdir(bbox_vehical_path)

                    bbox_p_v = list(set(bbox_pedestrian_lst).intersection(bbox_vehical_lst))
                    
                    tmp_vid = {"video": vid, "phases": []}

                    with open(caption_file_path, 'r') as file:
                        caption_data = json.load(file)["event_phase"]
                        for event in caption_data:
                            caption_description = []
                            caption_vehicle_list = event["caption_vehicle"].split(". ")
                            label = event["labels"][0]
                            tmp_phase = {}
                            tmp_phase["phase_number"] = label
                            if "positioned" in caption_vehicle_list[0]:
                                tmp_phase["caption"] = caption_vehicle_list[0]
                                tmp_phase["positions"] = []

                                for bb in bbox_p_v:
                                    f_bbox_ped = json.load(open(os.path.join(bbox_pedestrian_path, bb)))["annotations"]
                                    f_bbox_veh = json.load(open(os.path.join(bbox_vehical_path, bb)))["annotations"]

                                    tmp = {}

                                    for img in f_bbox_ped:
                                        if label[0] == img["phase_number"]:
                                            tmp["pedestrian"] = img["bbox"]

                                    for img in f_bbox_veh:
                                        if label[0] == img["phase_number"]:
                                            tmp["vehicle"] = img["bbox"]

                                    tmp_phase["positions"].append(tmp)

                            tmp_vid["phases"].append(tmp_phase)
                            # caption_data_process[event["labels"][0]] = {"description":
                            #                                             " ".join(caption_description if len(caption_description) > 0 else caption_pedestrian_list[0:2]),
                            #                                             "start_time": float(event["start_time"])}
                    position_analysis[phase].append(tmp_vid)
            except FileNotFoundError as e:
                print(e)
                continue



    with open(output_dir + '/position_analysis.json', 'w') as f:
        json.dump(position_analysis, f)


if __name__ == '__main__':
    """
        python ./analysis/caption_analysis.py
    """
    args = parse_args()
    ANALYSIS_DIR = args.analysis_dir
    OUTPUT_DIR = args.output_dir
    CAPTION_DIR = args.caption_dir
    BBOX_DIR = args.bbox_dir
    VIEW = str(args.view)
    NGRAM = args.ngram

    run_analysis(ANALYSIS_DIR, OUTPUT_DIR, CAPTION_DIR, BBOX_DIR, VIEW, NGRAM)
