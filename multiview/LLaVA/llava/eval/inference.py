import os
import json
import copy
import argparse
from tqdm import tqdm
from typing import Dict, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

import transformers

from llava.utils import disable_torch_init
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, \
                            MAX_VIDEO_LENGTH, NUM_CAMERA_VIEWS, IGNORE_INDEX
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path
)


def preprocess_multimodal(sources: Sequence[str],
                          config: transformers.PretrainedConfig) -> Dict:
    for source in sources:
        for sentence in source:
            if sentence["value"].startswith(DEFAULT_VIDEO_TOKEN):
                num_video_tokens = sentence["value"].count(DEFAULT_VIDEO_TOKEN)
                if num_video_tokens > MAX_VIDEO_LENGTH:
                    sentence["value"].replace(DEFAULT_VIDEO_TOKEN * num_video_tokens, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH)

            vid_replace_token = DEFAULT_IMAGE_TOKEN * MAX_VIDEO_LENGTH
            if config.mm_use_im_start_end:
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
    return sources


def preprocess_v1(sources,
                  tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if role == conv.roles[0]:
                conv.append_message(role, sentence["value"])
            elif role == conv.roles[1]:
                conv.append_message(role, None)
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations])

    return dict(input_ids=input_ids)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 video_folder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 video_processor,
                 model_config: transformers.PretrainedConfig,
                 ):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.video_folder = video_folder
    
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list" #FIXME

        if "videos" in sources[0]:
            source_id = self.list_data_dict[i]["id"]
            video_id, segment_id = source_id.rsplit("_", 1)

            video_files = self.list_data_dict[i]["videos"]
            video_paths = [os.path.join(self.video_folder, file) for file in video_files]
            images = [self.video_processor(path, return_tensors="pt")["pixel_values"][0] for path in video_paths]
            image_attention_masks = [torch.ones(image.shape[1], dtype=torch.bool) for image in images]

            assert len(images) > 0, f"Cannot open some videos with id: {sources[0]['id']}"
            for _ in range(len(images), NUM_CAMERA_VIEWS):
                images.append(torch.zeros(*images[0].shape))
                image_attention_masks.append(torch.zeros(images[0].shape[1], dtype=torch.bool))

            sources = preprocess_multimodal(copy.deepcopy([s["conversations"] for s in sources]), self.model_config)

            data_dict = preprocess_v1(sources, self.tokenizer)  

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0])

            if "videos" in self.list_data_dict[i]:
                data_dict["video_id"] = video_id
                data_dict["segment_id"] = segment_id
                data_dict["images"] = images
                data_dict["image_attention_masks"] = image_attention_masks

            return data_dict 
        

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        video_ids, segment_ids, input_ids = tuple([instance[key] for instance in instances] for key in ("video_id", "segment_id", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            video_ids=video_ids,
            segment_ids=segment_ids,
            input_ids=input_ids,
            # attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if all("images" in instance and instance["images"] for instance in instances):
            list_images = [instance["images"] for instance in instances]
            new_list_images = []
            for images in list_images:
                for image in images:
                    new_list_images.append(image)
            list_images = new_list_images     
            batch["images"] = list_images         
        else:
            raise ValueError(f"Image mismatched: {instances}")
        
        if all("image_attention_masks" in instance and instance["image_attention_masks"] for instance in instances):
            list_masks = [instance["image_attention_masks"] for instance in instances]
            new_list_masks = []
            for masks in list_masks:
                for mask in masks:
                    new_list_masks.append(mask)
            list_masks = new_list_masks
            batch["image_attention_masks"] = list_masks
        else:
            raise ValueError(f"Image attention masks mismatched: {instances}")
        
        return batch
        

def inference(args):
    disable_torch_init()

    results = {}
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, video_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.offload_folder
    )

    wts_dataset = LazySupervisedDataset(data_path=args.data_path,
                                        video_folder=args.video_folder,
                                        tokenizer=tokenizer,
                                        video_processor=video_processor,
                                        model_config=model.config)
    
    wts_datacollator = DataCollatorForSupervisedDataset(tokenizer)
    wts_dataloader = DataLoader(wts_dataset, collate_fn=wts_datacollator)

    with torch.inference_mode():
        for batch in tqdm(wts_dataloader):
            video_ids = batch["video_ids"]
            segment_ids = batch["segment_ids"]
            input_ids = batch["input_ids"]
            images = batch["images"]
            image_attention_masks = batch["image_attention_masks"]

            output_ids = model.generate(
                inputs=input_ids,
                images=images,
                image_attention_masks=image_attention_masks,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)
            # if video_id in results:
            #     results[video_id].append({
            #         "labels": [str(segment_id)],
            #         f"caption_{args.object}": outputs 
            #     })
            
            # else:
            #     results[video_id] = [{
            #         "labels": [str(segment_id)],
            #         f"caption_{args.object}": outputs
            #     }]
        
        # with open(args.output_path, "w") as f:
        #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--offload-folder", type=str, default=None)  
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    inference(args)