from .vid2seq import _get_tokenizer, Vid2Seq
from .texttitling import TextTilingTokenizer
from .llama_video import LlamaVideo

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2VisionModel
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig


def build_vid2seq_model(args, tokenizer):
    model = Vid2Seq(
        t5_path=args.model_name,
        num_features=args.max_feats,
        embed_dim=args.embedding_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        vis_drop=args.visual_encoder_dropout,
        enc_drop=args.text_encoder_dropout,
        dec_drop=args.text_decoder_dropout,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        label_smoothing=args.label_smoothing,
        use_speech=args.use_speech,
        use_video=args.use_video,
    )
    return model


def build_llama_video_model(args, tokenizer):
    vision_model = Blip2VisionModel.from_pretrained(
        args.llama_vision_encoder,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    llama_model = LlamaForCausalLM.from_pretrained(
        args.llama_language_model,
        cache_dir=args.cache_dir,
        device_map="auto",
        quantization_config=nf4_config,
        token="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",
    )

    return LlamaVideo(
        vision_encoder=vision_model,
        tokenizer=tokenizer,
        language_model=llama_model,
        freeze_vision_encoder=True,
        freeze_language_model=True,
    )
