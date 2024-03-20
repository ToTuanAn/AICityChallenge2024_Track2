import os
import cv2
import json
from tqdm import tqdm
import argparse
import regex as re
import numpy as np

# regex to find speed in sentence in format xx km/h
# example: he goes with the speed of 50 km/h in the highway
km_regex = re.compile(r'(\d+) km/h')
training_samples = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create speed prediction dataset from event video."
    )
    parser.add_argument(
        "--event_video_dir",
        type=str,
        required=True,
        help="Path to the event video directory",
    )
    parser.add_argument(
        "--event_label_path",
        type=str,
        required=True,
        help="Path to the event label file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,
        help="Number of frames to calculate average image",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step to take between frames",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=224,
        help="Height of the output image",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=224,
        help="Width of the output image",
    )
    return parser.parse_args()


def do_sampling(frames, sequence_length, step):
    if len(frames) < sequence_length:
        return [frames]
    
    frame_sequences = []
    for i in range(0, len(frames) - sequence_length + 1, step):
        frame_sequence = frames[i:i+sequence_length]
        frame_sequences.append(frame_sequence)
    return frame_sequences


def create_average_image(frames, weights=None):
    if weights is None:
        weights = [1] * len(frames)
        
    # normalize weights to sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # create average image
    average_image = np.zeros(frames[0].shape, dtype=np.float32)
    for frame, weight in zip(frames, weights):
        average_image += frame * weight
    
    # convert to uint8 and return
    average_image = average_image.astype(np.uint8)
    return average_image


if __name__ == "__main__":
    print("Creating dataset from event video...")
    
    # Parse arguments
    args = parse_args()
    event_video_dir = args.event_video_dir
    event_label_path = args.event_label_path
    output_dir = args.output_dir
    sequence_length = args.sequence_length
    step = args.step
    image_height = args.image_height
    image_width = args.image_width
    
    # Create output directory
    image_dir = os.path.join(output_dir, "images")
    label_path = os.path.join(output_dir, "labels.json")
    os.makedirs(image_dir, exist_ok=True)
    
    # Load event label
    with open(event_label_path, "r") as f:
        event_data = json.load(f)
    
    training_samples = []
    
    for event_name, event_data in tqdm(event_data.items(), total=len(event_data)):
        # Extract speed from sentence
        sentences = event_data['sentences']
        video_path = event_data['video_path']
        
        
        match = km_regex.search(sentences)
        if 'zero kilometers per hour'  in sentences or 'zero kilometer per hour' in sentences:
            speed = 0
        elif match:
            speed = int(match.group(1))
        else:
            print(f"Speed not found in {sentences}")
            continue
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_width, image_height))
            frames.append(frame)
        cap.release()
        
        frame_sequences = do_sampling(frames, args.sequence_length, args.step)
        for i, frame_sequence in enumerate(frame_sequences):
            average_image = create_average_image(frames)
            image_path = os.path.join(image_dir, f"{event_name}_seq{i}.jpg")
            cv2.imwrite(image_path, average_image)
            
            training_samples.append({
                "image_name": f"{event_name}_seq{i}.jpg",
                "speed": speed
            })
    
    print(f'Total training samples: {len(training_samples)}')
    with open(label_path, "w") as f:
        json.dump(training_samples, f, indent=4, ensure_ascii=False)
