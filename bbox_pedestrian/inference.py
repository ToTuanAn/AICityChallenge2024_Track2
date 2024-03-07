import cv2
from datasets import Dataset
from PIL import Image
import json
import os
from torch.utils.data import Dataset, DataLoader 
import torch

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="./data/test/public",
        help="Path to the directory containing WTS test",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bbox_pedestrian/output/test/public",
        help="Path to the output directory",
    )

    return parser.parse_args()



# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def load_datasets(path):
    datasets = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image)
            image_label = image.split(".")[0][-1]
            extention = image.split(".")[-1]
            if extention != "png":
                continue
            data = {}
            label = image.split("_")[-1][0]
            opencv_image=cv2.imread(image_path)
            opencv_image = maintain_aspect_ratio_resize(opencv_image, width = 224, height=224)
            color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            data["image"] = Image.fromarray(color_converted)
            data["id"] = folder
            data["label"] = image_label
            datasets.append(data)
    return datasets

class ImageCaptioningDataset(Dataset): 
    def __init__(self, dataset, processor): 
        self.dataset = dataset 
        self.processor = processor 
    def __len__(self): 
        return len(self.dataset) 
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt") 
        # remove batch dimension 
        encoding = {k: v.squeeze() for k, v in encoding.items()} 
        encoding["id"] = item["id"]
        return encoding 

def collate_fn(batch): 
    # pad the input_ids and attention_mask 
    processed_batch = {} 
    for key in batch[0].keys(): 
        if key != "id": 
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            processed_batch[key] = [example[key] for example in batch]

    return processed_batch

def inference(inference_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = VisionEncoderDecoderModel.from_pretrained("./model").to(device)
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.eval()
    batch_size = 16
    inference_datasets = load_datasets(inference_path)
    inference_datasets = ImageCaptioningDataset(inference_datasets, processor)
    inference_dataloader = DataLoader(inference_datasets, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    ids, labels, preds = [], [], []
    max_length = 50
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    for batch in inference_dataloader:
        pixel_values = batch.pop("pixel_values").to(device)
        
        outputs = model.generate(pixel_values=pixel_values, **gen_kwargs)
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ids.extend(batch.pop("id"))
        labels.extend(batch.pop("label"))

        preds.extend(pred)
    result = {}
    for i, l, p in zip(ids, labels, preds):
        result.setdefault(i, {})[l] = {"predict": p}
    with open(os.path.join(output_path, "output_pedestrian.json"), "w") as outfile: 
        json.dump(result, outfile, indent = 2)
    model.train()
        

                  
if __name__ == '__main__':
    """
        python ./bbox_pedestrian/preprocess.py
    """
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    INFERENCE_DIR = args.inference_dir


    inference(INFERENCE_DIR, OUTPUT_DIR)
