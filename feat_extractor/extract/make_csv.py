import pandas as pd
import glob
import os

VIDEO_PATH = "./data/videos"
FEATURE_PATH = "./data/features"
TRAIN_OR_VAL_CHOICE = ["train", "val"]
PREDESTRIAN_OR_VEHICLE_CHOICE = ["overhead_view", "vehicle_view"]

arr_jsonl = []

for TRAIN_OR_VAL in TRAIN_OR_VAL_CHOICE:
    for PREDESTRIAN_OR_VEHICLE in PREDESTRIAN_OR_VEHICLE_CHOICE:
        for video_subpath in glob.glob(f"{VIDEO_PATH}/{TRAIN_OR_VAL}/*"):
            for video in glob.glob(f"{video_subpath}/{PREDESTRIAN_OR_VEHICLE}/*.mp4"):
                output_path = f"{FEATURE_PATH}/{TRAIN_OR_VAL}/{video_subpath.split('/')[-1]}/{PREDESTRIAN_OR_VEHICLE}"
                os.makedirs(output_path, exist_ok=True)

                feature_path = f"{output_path}/{video.split('/')[-1][:-4]}.npy"
                arr_jsonl.append({"video_path" : video, "feature_path" : feature_path})

df = pd.DataFrame(arr_jsonl)
df.to_csv("./data/mapping_videopath_featurepath.csv", index=False)