import json
import argparse
from args import get_args_parser


def main(args):
    submission = {}
    pedestrian_preds = json.load(open(args.pedestrian, 'r'))
    vehicle_preds = json.load(open(args.vehicle, 'r'))

    for clip_id in pedestrian_preds:
        video_id, label = clip_id.split('#')
        pedestrian_caption = pedestrian_preds[clip_id]
        vehicle_caption = vehicle_preds[clip_id]

        if video_id not in submission:
            submission[video_id] = []

        submission[video_id].append(
            {
                'labels': [label],
                'caption_pedestrian': pedestrian_caption,
                'caption_vehicle': vehicle_caption,
            }
        )

    for video_id in submission:
        submission[video_id] = sorted(
            submission[video_id], key=lambda x: int(x['labels'][0]) * -1)

    with open(args.save, 'w') as f:
        json.dump(submission, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
