from transformers import AutoProcessor, Blip2ForConditionalGeneration
import os
import json
from tqdm import tqdm
from PIL import Image


type_caption = "pedestrian_caption_datasets"
def load_datasets(folder_dir, type_caption, image_dir="./data/video/", output_dir = './kaggle/working/image_datasets'):
    datasets = []
    for folder in tqdm(os.listdir(folder_dir)):
        folder_path = os.path.join(folder_dir, folder)
        for file in os.listdir(folder_path):
            print(file.replace(".json", "") != type_caption)
            if file.replace(".json", "") != type_caption:
                continue
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path) as file_o:
                    data = json.load(file_o)
                    for key, caption in data.items():
                        data = caption["caption"]
                        for vid in reversed(caption["videos"]):
                            try:
                                vid_name = vid.replace('.mp4', '')
                                for event in data:
                                    data_ = {}
                                    label = event['label']
                                    video_path = os.path.join(image_dir, folder, "avg"+ vid_name + f"phase{label}.png")
                                    data_["image"] = Image.open(video_path)
                                    data_["text"] = event["caption"]
                                    data_["id"] = vid_name + f"_{label}"
                                    datasets.append(data_)
                            except Exception as e:
                                print(e, video_path)
                                continue

            except Exception as e:
                print(e, video_path)
                continue
    return datasets    

def load_test_datasets(folder_dir, image_dir="./data/video/", output_dir = './kaggle/working/inference'):
    datasets = []
    for folder in tqdm(os.listdir(folder_dir)):
        if "external" in folder:
            folder_path = os.path.join(folder_dir, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path) as file_o:
                        data = json.load(file_o)
                        vid = data["video_name"]
                        vid_name = vid.replace('.mp4', '') 
                        data = data['event_phase']
                        for event in data:
                            label = event["labels"][0]
                            data_ = {}
                            video_path = os.path.join(image_dir, folder.replace("_external",""), "avg"+ vid_name + f"phase{label}.png")
                            data_["image"] = Image.open(video_path)
                            data_["text"] = ""
                            data_["id"] = vid_name + f"_{label}"
                            datasets.append(data_)
                              
                except Exception as e:
                    print(e, video_path)
                    continue
        else:
            folder_path = os.path.join(folder_dir, folder)
            for subfolder in os.listdir(folder_path):
                for view in ["vehicle_view", "overhead_view"]:
                    subfolder_path = os.path.join(folder_path, subfolder, view)
                    if not os.path.exists(subfolder_path):
                        continue
                    for file in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            with open(file_path) as file_o:
                                data = json.load(file_o)
                                if view == "vehicle_view":
                                    vid = data["vehicle_view"]
                                    vid_name = vid.replace('.mp4', '')
                                    data = data['event_phase']
                                    for event in data:
                                        data_ = {}
                                        label = event["labels"][0]
                                        video_path = os.path.join(image_dir, folder, "avg"+ vid_name + f"phase{label}.png")
                                        data_["image"] = Image.open(video_path)
                                        data_["text"] = ""
                                        data_["id"] = vid_name + f"_{label}"
                                        datasets.append(data_)
                                else:
                                    vids = data["overhead_videos"]
                                    data = data['event_phase']
                                    for vid in vids:
                                        vid_name = vid.replace('.mp4', '')
                                        for event in data:
                                            data_ = {}
                                            label = event["labels"][0]
                                            video_path = os.path.join(image_dir, folder, "avg"+ vid_name + f"phase{label}.png")
                                            data_["image"] = Image.open(video_path)
                                            data_["text"] = ""
                                            data_["id"] = vid_name + f"_{label}"
                                            datasets.append(data_)
                        except Exception as e:
                            print(e, video_path)
                            continue
    return datasets    

from torch.utils.data import Dataset, DataLoader 
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
        encoding["text"] = item["text"]
        return encoding 
def collate_fn(batch): 
    # pad the input_ids and attention_mask 
    processed_batch = {} 
    for key in batch[0].keys(): 
        if key != "id" and key != "text": 
            processed_batch[key] = torch.stack([example[key] for example in batch])
        elif key == "text":
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding="max_length", max_length=250, return_tensors="pt", truncation=True
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
        else:
            processed_batch[key] = [example[key] for example in batch]

    return processed_batch

import torch
def predict(model, test_dataloader, epoch=0):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"     
    ids, preds = [], []
    max_length = 250

    len_data = len(test_dataloader)
    for idx, batch in enumerate(test_dataloader): 
        pixel_values = batch.pop("pixel_values").to(device)
        
        outputs = model.generate(pixel_values=pixel_values, max_length=max_length)
        pred = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ids.extend(batch.pop("id"))
        preds.extend(pred)
    result = {}
    for i, p in zip(ids, preds):
        result[i] = p
    with open(f"predict_{type_caption}_{epoch}.json", "w") as outfile: 
        json.dump(result, outfile, indent = 2)
    model.train()
    return 

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("./output/pedestrian_caption_datasets_model_7", device_map="auto", load_in_8bit=True)
epochs = 10
lr = 1e-4
global_step = 0
batch_size = 16

train_dataset = load_datasets("./data/caption/output", type_caption)
train_dataset = ImageCaptioningDataset(train_dataset, processor) 
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
# test_datasets = load_test_datasets("./data/caption/test")

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

import wandb
wandb.login(key="8a97b2c8c0abcb6ac8879024b7812bf46d9e0542")
run = wandb.init(
    name=f"vit-gpt2-{type_caption}",
    project="caption-full",
    tags=["baseline"],
)

wandb.config = {"epochs": epochs, "learning_rate": lr, "batch_size": batch_size}

import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()
len_data = len(train_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len_data*epochs)

for epoch in range(epochs):
    total_loss = 0
    print("Epoch:", epoch)
    batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()

        optimizer.step()
        scheduler.step()
        batch_bar.update()
        del input_ids, pixel_values, outputs
        torch.cuda.empty_cache()
    total_loss /= len_data
    wandb.log({"loss": total_loss})
    model.save_pretrained(f"./output/{type_caption}_model_{epoch+3}")