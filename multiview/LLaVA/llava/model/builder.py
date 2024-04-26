#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN

from peft import PeftModel
from huggingface_hub import hf_hub_download


def load_from_hf(repo_id, filename, subfolder=None):
    cache_file = hf_hub_download(repo_id=repo_id,
                                 filename=filename,
                                 subfolder=subfolder)
    return torch.load(cache_file, map_location='cpu')


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if "llava" in model_name.lower():
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)


            print("DEBUG --- model", model)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                print("DEBUG --- lm_head empty")
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading mm_projector and multiview_ensembler weights...')
                
            mm_projector_state_dict = load_from_hf(model_path, "mm_projector.bin")
            multiview_ensembler_state_dict = load_from_hf(model_path, "multiview_ensembler.bin")
                    
            adapter_state_dict = {**mm_projector_state_dict, **multiview_ensembler_state_dict}
            adapter_state_dict = {(k[11:] if k.startswith('base_model.') else k): v for k, v in adapter_state_dict.items()}
            adapter_state_dict = {(k[6:] if k.startswith('model.model.') else k): v for k, v in adapter_state_dict.items()}
            
            for key in adapter_state_dict:
                print(f"{key} --- {adapter_state_dict[key].shape}")
            print(model.model.mm_projector[0].weight.shape)

            model.load_state_dict(adapter_state_dict, strict=False)

            print(model.model.mm_projector[0].weight.shape)


 
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    video_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        video_tower = model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            video_tower.to(device=device_map, dtype=torch.float16)
        else:
             video_tower.to(dtype=torch.float16)
        video_processor = video_tower.video_processor
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, video_processor, context_len