import json

f = open("./data/pedestrian_inference.json")
g = open("./data/pedestrian_inference.json")

pedestrian_data = json.load(f)
vehicle_data = json.load(g)
result_data = {}

for key in pedestrian_data["results"]:
    pedestrian_caption = pedestrian_data["results"][key]["sentence"]
    vehicle_caption = vehicle_data["results"][key]["sentence"]

    video_name, label = key.split("#")
    if video_name not in result_data:
        result_data[video_name] = []
    
    result_data[video_name].append({
        "labels" : [f"{label}"],
        "caption_pedestrian": pedestrian_caption,
        "caption_vehicle": vehicle_caption,
    })

with open("submission.json", "w") as outfile: 
    json.dump(result_data, outfile)
