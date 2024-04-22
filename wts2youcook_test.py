import os
import json


FPS = 30
annotation_dir = "../aic2024-t2/datasets/test"

all_annotation_files = []
all_video_names = []
for root, dirs, files in os.walk(annotation_dir):
    for file in files:
        if file.endswith(".json"):
            all_annotation_files.append(os.path.join(root, file))


all_annotation_data = {}

for filepath in sorted(all_annotation_files):
    print(f"Loading {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
        new_video_names = []
        if "BDD" in filepath:
            video_name = data["video_name"]
            all_video_names.append(video_name)  # for checking
            new_video_names.append(video_name)
        else:
            if "overhead" in filepath:
                all_video_names.extend(data["overhead_videos"])
                new_video_names.extend(data["overhead_videos"])
            else:
                all_video_names.append(data["vehicle_view"])
                new_video_names.append(data["vehicle_view"])

        for video_name in sorted(new_video_names):
            if video_name.endswith(".mp4"):
                video_name = video_name[:-4]

            fps = float(data.get("fps", FPS))
            duration = 0
            timestamps = []
            sentences = []
            events = data.get("event_phase", [])
            events = sorted(events, key=lambda x: float(x["labels"][0]))
            labels = [event["labels"][0] for event in events]
            # print(events)
            for event in events:
                start_time = float(event["start_time"])
                end_time = float(event["end_time"])
                # start = int(start_time * fps)
                # end = int(end_time * fps)
                start = start_time
                end = end_time
                timestamps.append((start, end))
                sentences.append("")

            all_annotation_data[video_name] = {
                "fps": fps,
                "duration": duration,
                "timestamps": timestamps,
                "sentences": sentences,
                "labels": labels,
            }

print(f"Loaded {len(all_annotation_files)} annotation files")
print(f"Loaded {len(all_video_names)} video names")

with open("annotations_video_intern/pedestrian_test.json", "w") as f:
    json.dump(all_annotation_data, f, indent=4)
with open("annotations_video_intern/vehicle_test.json", "w") as f:
    json.dump(all_annotation_data, f, indent=4)
