import json
import argparse

def main(args):
    results = {}
    pedestrian_results = json.load(open(args.pedestrian, "r"))
    vehicle_results = json.load(open(args.vehicle, "r"))

    for scenario_id in pedestrian_results:
        pedestrian_captions = sorted(pedestrian_results[scenario_id], key=lambda x: x["labels"][0], reverse=True)
        vehicle_captions = sorted(vehicle_results[scenario_id], key=lambda x: x["labels"][0], reverse=True)

        for pedestrian_caption, vehicle_caption in zip(pedestrian_captions, vehicle_captions):
            if scenario_id not in results:           
                results[scenario_id] = [{
                    "labels": pedestrian_caption["labels"],
                    "caption_pedestrian": pedestrian_caption["caption_pedestrian"],   
                    "caption_vehicle": vehicle_caption["caption_vehicle"],
                }]
            else:
                results[scenario_id].append({
                    "labels": pedestrian_caption["labels"],
                    "caption_pedestrian": pedestrian_caption["caption_pedestrian"],   
                    "caption_vehicle": vehicle_caption["caption_vehicle"],
                })

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pedestrian", type=str, default=None)
    parser.add_argument("--vehicle", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    main(args)