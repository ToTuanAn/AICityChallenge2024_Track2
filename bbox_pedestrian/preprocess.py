import cv2
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
        default="./data/caption",
        help="Path to the directory containing WTS caption",
    )

    parser.add_argument(
        "--bbox_dir",
        type=str,
        default="./data/bbox_annotated/pedestrian",
        help="Path to the directory containing WTS bbox",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        default="./data/videos",
        help="Path to the directory containing WTS video",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bbox_pedestrian/output",
        help="Path to the output directory",
    )

    parser.add_argument(
        "--view",
        type=View,
        default=View.Overhead,
        choices=list(View),
        help="Point of view of video to analyze",
    )

    return parser.parse_args()


def get_frame(vid_path, output_path, vid_name, phase, sec, bbox, extend_px=50):
    vid_cap = cv2.VideoCapture(vid_path)
    vid_cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid_cap.read()
    h, w, _ = image.shape
    bbox = [int(b) for b in bbox]
    if hasFrames:
        y_start = max(bbox[1] - extend_px, 0)
        y_end = min(bbox[1] + bbox[3] + extend_px, h)

        x_start = max(bbox[0] - extend_px, 0)
        x_end = min(bbox[0] + bbox[2] + extend_px, w)

        cropped_image = image[y_start: y_end, x_start:x_end]
        cv2.imwrite(f"{output_path}/{vid_name}_{phase}.png", cropped_image)
    return hasFrames

def preprocess(output_dir, caption_dir, bbox_dir, video_dir, view):
    phases = ["train", "val"]
    keyword = ["wear", "height", "wore", "dress", "worn"]

    for phase in phases:
        current_bbox_dir = os.path.join(bbox_dir, phase)
        current_caption_dir = os.path.join(caption_dir, phase)
        current_output_dir = os.path.join(output_dir, phase)
        current_video_dir = os.path.join(video_dir, phase)

        for vid in tqdm(os.listdir(current_bbox_dir)):
            bbox_path = os.path.join(current_bbox_dir, vid, view)
            caption_path = os.path.join(current_caption_dir, vid, view)
            output_path = os.path.join(current_output_dir, vid)
            video_path = os.path.join(current_video_dir, vid, view)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            try:
                caption_data_process = {}
                for caption_file in os.listdir(caption_path):
                    caption_file_path = os.path.join(caption_path, caption_file)
                    with open(caption_file_path, 'r') as file:
                        caption_data = json.load(file)["event_phase"]
                        for event in caption_data:
                            caption_description = []
                            caption_pedestrian_list = event["caption_pedestrian"].split(". ")
                            for caption_pedestrian in caption_pedestrian_list:
                                if any(word in caption_pedestrian for word in keyword):
                                    caption_description.append(caption_pedestrian)
                            caption_data_process[event["labels"][0]] = {"description":
                                                                        ". ".join(caption_description if len(caption_description) > 0 else caption_pedestrian_list[0:2]),
                                                                        "start_time": float(event["start_time"])}

                caption_output = {}
                bbox_labels = set()
                for bbox_file in os.listdir(bbox_path):
                    file_path = os.path.join(bbox_path, bbox_file)
                    vid_name = bbox_file.split("_bbox")[0]
                    vid_path = os.path.join(video_path, vid_name + ".mp4")
                    try:
                        with open(file_path, 'r') as file:
                            bbox_data = json.load(file)
                            bbox_data = bbox_data["annotations"]
                            for bbox in bbox_data:
                                bbox_labels.add(bbox["phase_number"])
                                get_frame(vid_path, output_path, vid_name, bbox["phase_number"], caption_data_process[bbox["phase_number"]]["start_time"], bbox["bbox"])
                    except:
                        continue

                for bbox in bbox_labels:
                    caption_output[bbox] = caption_data_process[bbox]["description"]
                    caption_output_path = os.path.join(output_path, f"{vid}_caption.json")
                    with open(caption_output_path, 'w') as file:
                        json.dump(caption_output, file, indent=2)
            except:
                # print(e)
                continue
    print("success")
                            
if __name__ == '__main__':
    """
        python ./bbox_pedestrian/preprocess.py
    """
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    CAPTION_DIR = args.caption_dir
    VIDEO_DIR = args.video_dir
    BBOX_DIR = args.bbox_dir

    VIEW = str(args.view)


    preprocess(OUTPUT_DIR, CAPTION_DIR, BBOX_DIR, VIDEO_DIR, VIEW)
