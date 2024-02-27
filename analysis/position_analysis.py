import os
import argparse
import enum
import json
import string
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from operator import sub

# CCW : Counter-Clock-Wise
# vector subtraction : tuple(map(sub, p1, p2)) 
def ccw(a, b, c):
    # a = [a[0] + a[2] / 2, a[1] + a[3] / 2]
    # b = [b[0] + b[2] / 2, b[1] + b[3] / 2]
    # c = [c[0] + c[2] / 2, c[1] + c[3] / 2]
    a, c = tuple(map(sub, b, a)), tuple(map(sub, c, a))
    return a[0]*c[1]-a[1]*c[0]

def angle(a, b, c):
    """
        a is corner
    """

    a = [a[0] + a[2] / 2, a[1] + a[3] / 2]
    b = [b[0] + b[2] / 2, b[1] + b[3] / 2]
    c = [c[0] + c[2] / 2, c[1] + c[3] / 2]
    def dist(a):
        return np.sqrt([(a[0])**2 + (a[1])**2])[0]
    a = (a[0] - b[0], a[1] - b[1])
    b = (a[0] - c[0], a[1] - c[1])

    return np.arccos((a[0]*b[0] + a[1]*b[1])/(dist(a) * dist(b)))


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

    return parser.parse_args()


def check_have_location(caption_lst: list, keywords: list[str] = ["positioned", "located", "situated"]):
    for keyword in keywords:
        for index, cap in enumerate(caption_lst):
            if keyword in cap and "vehicle" in cap:
                return True, index
            
    return False, None

def check_true(ccw_value, category):
    if category == "left":
        return ccw_value < 0
    elif category == "right":
        return ccw_value > 0

    return category


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

                            _have, _idx = check_have_location(caption_vehicle_list)
                            if _have:
                                tmp_phase["caption"] = caption_vehicle_list[_idx]
                                if "right" in tmp_phase["caption"]:
                                    tmp_phase["category"] = "right"
                                elif "left" in tmp_phase["caption"]:
                                    tmp_phase["category"] = "left"
                                elif "in front of" in tmp_phase["caption"]:
                                    tmp_phase["category"] = "in front of"
                                elif "behind" in tmp_phase["caption"]:
                                    tmp_phase["category"] = "behind"

                                tmp_phase["positions"] = []

                                for bb in bbox_p_v:
                                    f_bbox_ped = json.load(open(os.path.join(bbox_pedestrian_path, bb)))["annotations"]
                                    f_bbox_veh = json.load(open(os.path.join(bbox_vehical_path, bb)))["annotations"]

                                    tmp = {
                                        "camera": bb
                                    }

                                    for img in f_bbox_ped:
                                        if label[0] == img["phase_number"]:
                                            tmp["pedestrian"] = img["bbox"]
                                        
                                        if str(int(label[0]) + 1) == img["phase_number"]:
                                            tmp["pedestrian_next"] = img["bbox"]

                                        if str(int(label[0]) - 1) == img["phase_number"]:
                                            tmp["pedestrian_prev"] = img["bbox"]

                                    for img in f_bbox_veh:
                                        if label[0] == img["phase_number"]:
                                            tmp["vehicle"] = img["bbox"]

                                    if "pedestrian" in tmp.keys() and "pedestrian_prev" in tmp.keys() and "vehicle" in tmp.keys():
                                        ccw_value = ccw(tmp["pedestrian_prev"], tmp["pedestrian"], tmp["vehicle"])
                                        tmp["prev_ccw"] = [
                                            ccw_value,
                                            check_true(ccw_value, tmp_phase["category"]) if "category" in tmp_phase.keys() else None,
                                            # angle(tmp["pedestrian"], tmp["pedestrian_prev"], tmp["vehicle"])
                                        ]
                                    if "pedestrian" in tmp.keys() and "pedestrian_next" in tmp.keys() and "vehicle" in tmp.keys():
                                        ccw_value = ccw(tmp["pedestrian"], tmp["pedestrian_next"], tmp["vehicle"])
                                        tmp["next_ccw"] = [
                                            ccw_value,
                                            check_true(ccw_value, tmp_phase["category"]) if "category" in tmp_phase.keys() else None,
                                            # angle(tmp["pedestrian_next"], tmp["pedestrian"], tmp["vehicle"])
                                        ]

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
    

def run_eda(output_dir):
    f = open(output_dir + '/position_analysis.json', 'r')
    position_analysis = json.load(f)

    cnt = {
        "0": {
            "prev": 0,
            "next": 0
        },
        "1": {
            "prev": 0,
            "next": 0
        },
        "2": {
            "prev": 0,
            "next": 0
        },
        "3": {
            "prev": 0,
            "next": 0
        },
        "4": {
            "prev": 0,
            "next": 0
        }
    }
    _cnt = {
        "0": {
            "true_prev": 0,
            "true_next": 0,
            "true_p_wo_n": 0,
            "true_n_wo_q": 0,
        },
        "1": {
            "true_prev": 0,
            "true_next": 0,
            "true_p_wo_n": 0,
            "true_n_wo_q": 0,
        },
        "2": {
            "true_prev": 0,
            "true_next": 0,
            "true_p_wo_n": 0,
            "true_n_wo_q": 0,
        },
        "3": {
            "true_prev": 0,
            "true_next": 0,
            "true_p_wo_n": 0,
            "true_n_wo_q": 0,
        },
        "4": {
            "true_prev": 0,
            "true_next": 0,
            "true_p_wo_n": 0,
            "true_n_wo_q": 0,
        }
    }

#209 96
    for vid_info in position_analysis["train"]:
        for phase in vid_info["phases"]:
            if "caption" in phase.keys() and ("right" in phase["caption"] or "left" in phase["caption"]):
                phase_num = phase["phase_number"]
                for pos in phase["positions"]:
                    if "next_ccw" in pos.keys():
                        cnt[phase_num]["next"] += 1
                        _cnt[phase_num]["true_next"] += int(pos["next_ccw"][-1])
                        if not pos["next_ccw"][-1]:
                            print(pos["next_ccw"][0])
                        # if "prev_ccw" in pos.keys():
                        #     _cnt[phase_num]["true_n_wo_q"] += int(pos["prev_ccw"][-1])
                    
                    if "prev_ccw" in pos.keys():
                        cnt[phase_num]["prev"] += 1
                        _cnt[phase_num]["true_prev"] += int(pos["prev_ccw"][-1])
                        # if "next_ccw" in pos.keys():
                        #     _cnt[phase_num]["true_p_wo_n"] += int(pos["next_ccw"][-1])

    print(cnt)
    print(_cnt)


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
    run_eda(OUTPUT_DIR)
