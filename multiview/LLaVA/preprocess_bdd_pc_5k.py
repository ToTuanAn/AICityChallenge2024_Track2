import os
import json
import argparse
from tqdm import tqdm


INSTRUCTONS = {
    "pedestrian": "<video>\nDescribe the traffic safety scenario in the video. The caption must contain details about the surrounding context, attention, location, and behavior of the pedestrian.",
    "vehicle": "<video>\nDescribe the traffic safety scenario in the video. The caption must contain details about the surrounding context, attention, location, and behavior of the vehicle."
}


def trim_video(video_path, video_output_path, start_time, end_time):
    video_output_folder = video_output_path.replace("\\", "/").replace("//", "/").strip("/").rsplit("/", 1)[0]
    os.makedirs(video_output_folder, exist_ok=True)
    cmd = f"ffmpeg -ss {start_time} -to {end_time} -i {video_path} -ss {start_time} -to {end_time} -copyts -filter:v fps=10 {video_output_path}"
    os.system(cmd)


def create_sources_and_trim_video(annotation_path, video_path, video_output_folder, object_type):
    sources = {}
    scenario_id = video_path.replace("\\", "/").replace("//", "/").strip("/").split("/")[-1][:-4]
    annotation = json.load(open(annotation_path, "r"))

    for video_file in [annotation["video_name"]]:
        for i in annotation["event_phase"]:
            label = i["labels"][0]
            start_time = i["start_time"]
            end_time = i["end_time"]
    
            video_output_file = f"{scenario_id}/phase_{label}/{video_file[:-4]}_{start_time}_{end_time}.mp4"
            video_output_path = os.path.join(video_output_folder, video_output_file)

            trim_video(video_path, video_output_path, start_time, end_time)

            if label not in sources:
                sources[label] = {
                    "id": f"{scenario_id}_phase_{label}",
                    "videos": [
                        video_output_file,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": INSTRUCTONS[object_type],
                        },
                        {
                            "from": "gpt",
                            "value": i[f"caption_{object_type}"],
                        },
                    ],
                }
            else:
                sources[label]["videos"].append(video_output_file)

    sources = [sources[k] for k in sources]
    return sources, scenario_id


def main(args):
    for split in ["train", "val"]:
        # for scenario_type in ["", "normal_trimmed"]:
            # if scenario_type:    
            #     print(f"Preprocessing {split} split --- mixed cases...")
            # else:
            #     print(f"Preprocessing {split} split --- normal cases...")

            print(f"Preprocessing {split}")

            annotation_folder = os.path.join(args.annotation_folder, f"caption/{split}")
            video_folder = os.path.join(args.video_folder, f"{split}/vehicle_view")
            
            annotation_files = os.listdir(annotation_folder)
            video_files = os.listdir(video_folder)

            annotation_paths = [os.path.join(annotation_folder, annotation_file) for annotation_file in annotation_files]
            video_paths = [os.path.join(video_folder, video_file) for video_file in video_files]
             
            for annotation_path, video_path in tqdm(zip(annotation_paths, video_paths)):
                for object_type in ["pedestrian", "vehicle"]:
                    sources, scenario_id = create_sources_and_trim_video(annotation_path, video_path, args.video_output_folder, object_type)
                    
                    # DEBUG
                    save_folder = os.path.join(args.annotation_output_folder, object_type)
                    save_path = os.path.join(save_folder, f"{scenario_id}.json")
                    os.makedirs(save_folder, exist_ok=True)
                    with open(save_path, "w") as f:
                        json.dump(sources, f, indent=4)        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default=None)
    parser.add_argument("--annotation-output-folder", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default=None)
    parser.add_argument("--video-output-folder", type=str, default=None)
    args = parser.parse_args()

    main(args)