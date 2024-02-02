import os
import cv2
import json
from copy import deepcopy
import argparse


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
    return parser.parse_args()


def convert_wts_to_youcook_format(video_filepath, caption_filepath):
    print(f"Loading video: {video_filepath}")
    cap = cv2.VideoCapture(video_filepath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 10
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    print(f" + Duration: {duration}s")
    print(f" + FPS: {fps}")

    print(f"Reading caption: {caption_filepath}")
    with open(caption_filepath, "r") as f:
        caption_data = json.load(f)
    event_phase = caption_data["event_phase"]
    event_phase = sorted(event_phase, key=lambda x: x["labels"][0])

    pedestrian_data = {
        "duration": round(duration, 2),
        "fps": 1,
        "original_fps": round(fps, 2),
        "timestamps": [],
        "sentences": [],
    }
    vehicle_data = deepcopy(pedestrian_data)
    for event in event_phase:
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])
        caption_pedestrian = event["caption_pedestrian"]
        caption_vehicle = event["caption_vehicle"]

        start = round(start_time * fps)
        end = round(end_time * fps)

        pedestrian_data["timestamps"].append([start, end])
        pedestrian_data["sentences"].append(caption_pedestrian)

        vehicle_data["timestamps"].append([start, end])
        vehicle_data["sentences"].append(caption_vehicle)

    # print("pedestrian data:")
    # print(json.dumps(pedestrian_data, indent=4))

    # print("Vehicle data:")
    # print(json.dumps(vehicle_data, indent=4))
    # exit()

    return pedestrian_data, vehicle_data


if __name__ == "__main__":
    args = parse_args()
    VIDEOS_DIR = args.videos_dir
    CAPTION_DIR = args.caption_dir
    # BBOX_DIR = args.bbox_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in sorted(os.listdir(CAPTION_DIR)):
        videos_split_dir = os.path.join(VIDEOS_DIR, split)
        caption_split_dir = os.path.join(CAPTION_DIR, split)
        
        split_annotation_data = {
            "pedestrian": {},
            "vehicle": {},
        }
        
        for video_id in sorted(os.listdir(caption_split_dir)):
            print(f"Processing {video_id} ...")
            video_dir = os.path.join(videos_split_dir, video_id)
            caption_dir = os.path.join(caption_split_dir, video_id)
            
            if video_id != "normal_trimmed":
                for view in sorted(os.listdir(caption_dir)):
                    video_view_dir = os.path.join(video_dir, view)
                    caption_view_dir = os.path.join(caption_dir, view)
                    
                    for json_file in os.listdir(caption_view_dir):
                        if not json_file.endswith(".json"):
                            continue
                        
                        json_filepath = os.path.join(caption_view_dir, json_file)
                        print("Processing", json_filepath)
                        with open(json_filepath, "r") as f:
                            caption_data = json.load(f)
                        
                        video_filenames = []
                        if view.lower() == "overhead_view":
                            video_filenames = caption_data['overhead_videos']
                        else:
                            video_filenames.append(caption_data['vehicle_view'])

                        for video_filename in video_filenames:
                            video_filepath = os.path.join(video_view_dir, video_filename)
                            if not video_filename.lower().endswith(".mp4"):
                                print("Skipping", video_filepath)
                                continue
                            
                            print("Processing", video_filepath)
                            pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
                                video_filepath, json_filepath
                            )
                            video_name = video_filename.split(".")[:-1]
                            video_name = ".".join(video_name)
                            split_annotation_data["pedestrian"][video_name] = pedestrian_data
                            split_annotation_data["vehicle"][video_name] = vehicle_data
            else:
                for normal_video_id in sorted(os.listdir(video_dir)):
                    normal_video_dir = os.path.join(video_dir, normal_video_id)
                    normal_caption_dir = os.path.join(caption_dir, normal_video_id)
                    
                    for normal_view in sorted(os.listdir(normal_caption_dir)):
                        normal_video_view_dir = os.path.join(normal_video_dir, normal_view)
                        normal_caption_view_dir = os.path.join(normal_caption_dir, normal_view)
                        
                        for json_file in os.listdir(normal_caption_view_dir):
                            if not json_file.endswith(".json"):
                                continue
                            
                            json_filepath = os.path.join(normal_caption_view_dir, json_file)
                            print("Processing", json_filepath)
                            with open(json_filepath, "r") as f:
                                caption_data = json.load(f)
                            
                            video_filenames = []
                            if normal_view.lower() == "overhead_view":
                                video_filenames = caption_data['overhead_videos']
                            else:
                                video_filenames.append(caption_data['vehicle_view'])

                            for video_filename in video_filenames:
                                video_filepath = os.path.join(normal_video_view_dir, video_filename)
                                if not video_filename.lower().endswith(".mp4"):
                                    print("Skipping", video_filepath)
                                    continue
                                
                                print("Processing", video_filepath)
                                pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
                                    video_filepath, json_filepath
                                )
                                video_name = video_filename.split(".")[:-1]
                                video_name = ".".join(video_name)
                                split_annotation_data["pedestrian"][video_name] = pedestrian_data
                                split_annotation_data["vehicle"][video_name] = vehicle_data
        
        for obj in ["pedestrian", "vehicle"]:
            output_filepath = os.path.join(OUTPUT_DIR, f"{obj}_{split}.json")
            with open(output_filepath, "w") as f:
                json.dump(split_annotation_data[obj], f, indent=4)
        

    # for split in sorted(os.listdir(VIDEOS_DIR)):
    #     videos_split_dir = os.path.join(VIDEOS_DIR, split)
    #     caption_split_dir = os.path.join(CAPTION_DIR, split)
    #     # bbox_split_dir = os.path.join(BBOX_DIR, split)

    #     split_annotation_data = {
    #         "pedestrian": {},
    #         "vehicle": {},
    #     }

    #     if not os.path.isdir(videos_split_dir):
    #         continue

    #     for video_folder in sorted(os.listdir(videos_split_dir)):
    #         print(f"Processing {video_folder} ...")
    #         video_dir = os.path.join(videos_split_dir, video_folder)
    #         if not os.path.isdir(video_dir):
    #             continue
            
    #         if video_folder == "normal_trimmed":
    #             for normal_video_folder in sorted(os.listdir(video_dir)):
    #                 normal_video_dir = os.path.join(video_dir, normal_video_folder)
    #                 if not os.path.isdir(normal_video_dir):
    #                     continue
    #                 for normal_view in sorted(os.listdir(normal_video_dir)):
    #                     normal_view_dir = os.path.join(normal_video_dir, normal_view)
    #                     if not os.path.isdir(normal_view_dir):
    #                         continue
    #                     for normal_video_filename in sorted(os.listdir(normal_view_dir)):
    #                         normal_video_filepath = os.path.join(normal_view_dir, normal_video_filename)
    #                         if not normal_video_filename.lower().endswith(".mp4"):
    #                             print("Skipping", normal_video_filepath)
    #                             continue
    #                         print("Processing", normal_video_filepath)
    #                         caption_filename = normal_video_filename.split(".")[:-1]
    #                         caption_filename = '.'.join(caption_filename)
    #                         caption_filename += "_caption.json"
    #                         caption_filepath = os.path.join(
    #                             caption_split_dir, video_folder, normal_video_folder, normal_view, caption_filename
    #                         )
    #                         print("Caption file:", caption_filepath)
    #                         pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
    #                             normal_video_filepath, caption_filepath
    #                         )
    #                         video_name = normal_video_filename.split(".")[:-1]
    #                         video_name = ".".join(video_name)
    #                         split_annotation_data["pedestrian"][normal_video_folder] = pedestrian_data
    #                         split_annotation_data["vehicle"][normal_video_folder] = vehicle_data
    #         else:
    #             # continue
    #             for view in sorted(os.listdir(video_dir)):
    #                 view_dir = os.path.join(video_dir, view)
    #                 if not os.path.isdir(view_dir):
    #                     continue

    #                 # if view.lower() == "overhead_view":
    #                 #     is_pedestrian = True
    #                 # else:
    #                 #     is_pedestrian = False

    #                 for video_filename in sorted(os.listdir(view_dir)):
    #                     video_filepath = os.path.join(view_dir, video_filename)
    #                     if not video_filename.lower().endswith(".mp4"):
    #                         print("Skipping", video_filepath)
    #                         continue

    #                     print("Processing", video_filepath)
    #                     caption_filename = video_filename.split(".")[:-1]
    #                     caption_filename = '.'.join(caption_filename)
    #                     caption_filename = caption_filename.split("_")[:4]
    #                     caption_filename += ["caption"]
    #                     caption_filename = "_".join(caption_filename) + ".json"
    #                     caption_filepath = os.path.join(
    #                         caption_split_dir, video_folder, view, caption_filename
    #                     )
    #                     print("Caption file:", caption_filepath)

    #                     pedestrian_data, vehicle_data = convert_wts_to_youcook_format(
    #                         video_filepath, caption_filepath
    #                     )

    #                     video_name = video_filename.split(".")[:-1]
    #                     video_name = ".".join(video_name)
    #                     split_annotation_data["pedestrian"][video_folder] = pedestrian_data
    #                     split_annotation_data["vehicle"][video_folder] = vehicle_data
    #         #         break
    #         #     break

    #         # split_annotation_data["pedestrian"][video_folder] = perdestrian_data
    #         # split_annotation_data["vehicle"][video_folder] = vehicle_data

        for obj in ["pedestrian", "vehicle"]:
            output_filepath = os.path.join(OUTPUT_DIR, f"{obj}_{split}.json")
            with open(output_filepath, "w") as f:
                json.dump(split_annotation_data[obj], f, indent=4)
