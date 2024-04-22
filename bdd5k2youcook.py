import os
import cv2
import json
from copy import deepcopy
import argparse

event_sort_key = {
    "prerecognition": 0,
    "recognition": 1,
    "judgement": 2,
    "action": 3,
    "avoidance": 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="./datasets/videos",
        required=True,
        help="Path to the directory containing WTS videos",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="./datasets/annotations/caption",
        required=True,
        help="Path to the directory containing WTS captions",
    )
    parser.add_argument("--bbox_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/new_annotations",
    )
    parser.add_argument("--merge_val", action="store_true")
    return parser.parse_args()


def convert_bdd5k_to_youcook_format(video_filepath, caption_filepath):
    # print(f"Loading video: {video_filepath}")
    cap = cv2.VideoCapture(video_filepath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    if video_filepath.split("/")[-3] == "20231006_21_CN5_T1":
        print("Large videos: ", video_filepath)
        fps = 10

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps

    with open(caption_filepath, "r") as f:
        caption_data = json.load(f)
    event_phase = caption_data["event_phase"]
    event_phase = sorted(event_phase, key=lambda x: event_sort_key[x["labels"][0]])
    video_fps = float(
        caption_data.get("fps", video_fps)
    )  # use the fps from the caption file if available
    if video_fps > 40:
        video_fps = 30

    pedestrian_data = {
        "duration": round(duration, 2),
        "fps": 1,
        "original_fps": round(fps, 2),
        "timestamps": [],
        "sentences": [],
        "video_fps": round(video_fps, 2),
        "timestamps_video_fps": [],
    }
    vehicle_data = deepcopy(pedestrian_data)
    for event in event_phase:
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])
        caption_pedestrian = event["caption_pedestrian"]
        caption_vehicle = event["caption_vehicle"]

        # start = round(start_time * fps)
        # end = round(end_time * fps)
        start = start_time
        end = end_time

        start_video_fps = round(start_time * video_fps)
        end_video_fps = round(end_time * video_fps)

        pedestrian_data["timestamps"].append([start, end])
        pedestrian_data["timestamps_video_fps"].append([start_video_fps, end_video_fps])
        pedestrian_data["sentences"].append(caption_pedestrian)

        vehicle_data["timestamps"].append([start, end])
        vehicle_data["timestamps_video_fps"].append([start_video_fps, end_video_fps])
        vehicle_data["sentences"].append(caption_vehicle)

    return pedestrian_data, vehicle_data


if __name__ == "__main__":
    args = parse_args()
    VIDEOS_DIR = args.videos_dir
    CAPTION_DIR = args.caption_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annotation_data = {}

    for split in sorted(os.listdir(CAPTION_DIR)):
        if split not in ["train", "val"]:
            continue
        videos_split_dir = os.path.join(VIDEOS_DIR, split)
        caption_split_dir = os.path.join(CAPTION_DIR, split)

        split_annotation_data = {
            "pedestrian": {},
            "vehicle": {},
        }

        print(f"Processing {split} ...")

        for view in sorted(os.listdir(caption_split_dir)):
            video_view_dir = os.path.join(videos_split_dir, view)
            caption_view_dir = os.path.join(caption_split_dir, view)

            if not os.path.isdir(video_view_dir) or not os.path.isdir(caption_view_dir):
                print(f"Skipping {view} ...")
                continue

            print(f"Processing {view} ...")

            for json_file in os.listdir(caption_view_dir):
                if not json_file.endswith(".json"):
                    continue

                json_filepath = os.path.join(caption_view_dir, json_file)
                # print("Processing", json_filepath)
                with open(json_filepath, "r") as f:
                    caption_data = json.load(f)

                video_name = caption_data["video_name"]
                video_path = os.path.join(video_view_dir, video_name)

                if not os.path.exists(video_path):
                    print("Video not found:", video_path)
                    exit(1)

                pedestrian_data, vehicle_data = convert_bdd5k_to_youcook_format(
                    video_path, json_filepath
                )

                pedestrian_data["video_path"] = video_path
                vehicle_data["video_path"] = video_path

                if video_name.endswith(".mp4"):
                    video_name = ".".join(video_name.split(".")[:-1])
                split_annotation_data["pedestrian"][video_name] = pedestrian_data
                split_annotation_data["vehicle"][video_name] = vehicle_data

        annotation_data[split] = split_annotation_data

    if args.merge_val:
        for video in annotation_data["val"]["pedestrian"]:
            annotation_data["train"]["pedestrian"][video] = annotation_data["val"][
                "pedestrian"
            ][video]
            annotation_data["train"]["vehicle"][video] = annotation_data["val"][
                "vehicle"
            ][video]

    for split in annotation_data:
        for obj in ["pedestrian", "vehicle"]:
            output_filepath = os.path.join(OUTPUT_DIR, f"{obj}_{split}.json")
            with open(output_filepath, "w") as f:
                json.dump(annotation_data[split][obj], f, indent=4)

        # for video_id in sorted(os.listdir(caption_split_dir)):
        #     # print(f"Processing {video_id} ...")
        #     video_dir = os.path.join(videos_split_dir, video_id)
        #     caption_dir = os.path.join(caption_split_dir, video_id)

        #     if video_id == "normal_trimmed":
        #         for normal_video_id in sorted(os.listdir(video_dir)):
        #             normal_video_dir = os.path.join(video_dir, normal_video_id)
        #             normal_caption_dir = os.path.join(caption_dir, normal_video_id)

        #             for normal_view in sorted(os.listdir(normal_caption_dir)):
        #                 normal_video_view_dir = os.path.join(normal_video_dir, normal_view)
        #                 normal_caption_view_dir = os.path.join(normal_caption_dir, normal_view)

        #                 for json_file in os.listdir(normal_caption_view_dir):
        #                     if not json_file.endswith(".json"):
        #                         continue

        #                     json_filepath = os.path.join(normal_caption_view_dir, json_file)
        #                     # print("Processing", json_filepath)
        #                     with open(json_filepath, "r") as f:
        #                         caption_data = json.load(f)

        #                     video_filenames = []
        #                     if normal_view.lower() == "overhead_view":
        #                         video_filenames = caption_data['overhead_videos']
        #                     else:
        #                         video_filenames.append(caption_data['vehicle_view'])

        #                     for video_filename in video_filenames:
        #                         video_filepath = os.path.join(normal_video_view_dir, video_filename)
        #                         if not video_filename.lower().endswith(".mp4"):
        #                             print("Skipping", video_filepath)
        #                             continue

        #                         # print("Processing", video_filepath)
        #                         pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
        #                             video_filepath, json_filepath
        #                         )
        #                         pedestrian_data["video_path"] = video_filepath
        #                         vehicle_data["video_path"] = video_filepath
        #                         video_name = video_filename.split(".")[:-1]
        #                         video_name = ".".join(video_name)
        #                         split_annotation_data["pedestrian"][video_name] = pedestrian_data
        #                         split_annotation_data["vehicle"][video_name] = vehicle_data
        #     else:
        #         for view in sorted(os.listdir(caption_dir)):
        #             video_view_dir = os.path.join(video_dir, view)
        #             caption_view_dir = os.path.join(caption_dir, view)

        #             for json_file in os.listdir(caption_view_dir):
        #                 if not json_file.endswith(".json"):
        #                     continue

        #                 json_filepath = os.path.join(caption_view_dir, json_file)
        #                 # print("Processing", json_filepath)
        #                 with open(json_filepath, "r") as f:
        #                     caption_data = json.load(f)

        #                 video_filenames = []
        #                 if view.lower() == "overhead_view":
        #                     video_filenames = caption_data['overhead_videos']
        #                 else:
        #                     video_filenames.append(caption_data['vehicle_view'])

        #                 for video_filename in video_filenames:
        #                     video_filepath = os.path.join(video_view_dir, video_filename)
        #                     if not video_filename.lower().endswith(".mp4"):
        #                         print("Skipping", video_filepath)
        #                         continue

        #                     # print("Processing", video_filepath)
        #                     pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
        #                         video_filepath, json_filepath
        #                     )
        #                     pedestrian_data["video_path"] = video_filepath
        #                     vehicle_data["video_path"] = video_filepath
        #                     video_name = video_filename.split(".")[:-1]
        #                     video_name = ".".join(video_name)
        #                     split_annotation_data["pedestrian"][video_name] = pedestrian_data
        #                     split_annotation_data["vehicle"][video_name] = vehicle_data

        # for obj in ["pedestrian", "vehicle"]:
        #     output_filepath = os.path.join(OUTPUT_DIR, f"{obj}_{split}.json")
        #     with open(output_filepath, "w") as f:
        #         json.dump(split_annotation_data[obj], f, indent=4)
