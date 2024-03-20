import os
import json
import regex as re

vehicle_label_path = '../new_annotations/vehicle_train.json'

with open(vehicle_label_path, 'r') as f:
    vehicle_label = json.load(f)

# regex to find speed in sentence in format xx km/h
# example: he goes with the speed of 50 km/h in the highway
km_regex = re.compile(r'(\d+) km/h')
training_samples = []

for video_name, video_data in vehicle_label.items():
    if "vehicle_view" not in video_name:
        continue
        
    video_path = video_data['video_path']
    sentences = video_data['sentences']
    timestamps = video_data['timestamps']
    for i, (sentence, timestamp) in enumerate(zip(sentences, timestamps)):
        match = km_regex.search(sentence)
        if match:
            print(match.group(1) + " km/h")
            training_samples.append({
                "video_name": video_name,
                "video_path": video_path,
                "timestamp": timestamp,
                "speed": float(match.group(1)),
                "sentence": sentence,
                "event_index": i,
            })
            
print(f'Total training samples: {len(training_samples)}')
with open('training_samples.json', 'w') as f:
    json.dump(training_samples, f, indent=4, ensure_ascii=False)