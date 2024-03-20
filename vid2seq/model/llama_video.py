import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2VisionModel
from transformers import LlamaTokenizer, LlamaForCausalLM


class LlamaVideo(nn.Module):
    def __init__(
        self,
        vision_encoder,
        tokenizer,
        language_model,
        freeze_vision_encoder=True,
        freeze_language_model=True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer
        self.lm = language_model

        vision_encoder_output_dim = vision_encoder.config.hidden_size
        lm_output_dim = language_model.config.hidden_size
        self.vision_projection = nn.Linear(768, lm_output_dim)
        if freeze_vision_encoder:
            self.freeze_vision_encoder()
        if freeze_language_model:
            self.freeze_language_model()

    def freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("Freeze vision encoder")

    def freeze_language_model(self):
        for param in self.lm.parameters():
            param.requires_grad = False
        print("Freeze language model")

    @torch.autocast(device_type="cuda")
    def forward(self, video, input_tokenized, output_tokenized):
        """Forward pass of VideoLlama

        Args:
            frames (torch.Tensor): Input in shape (batch_size, seq_len, channels, width, height) or (batch_size, seq_len, features)
            tokenized_input (dict): input dictionary of the prompts
        """
        if isinstance(video, dict):  # cached
            video, atts_vis = video["video"], video["atts_vis"]
        else:
            if (
                len(video.size()) == 5
            ):  # [batch_size, seq_length, channels, width, height]
                frame_features = []
                for batch_idx in range(video.size(0)):
                    cur_frames = video[batch_idx].contiguous()
                    frame_features.append(
                        self.vision_encoder(cur_frames).last_hidden_state[:, 0, :]
                    )
                frame_features = torch.stack(frame_features)
                frame_features = self.vision_projection(frame_features)
            elif len(video.size()) == 3:  # [batch_size, seq_length, features]
                frame_features = self.vision_projection(video)
            else:
                raise ValueError(
                    "Invalid input shape. Must be either (batch_size, seq_len, channels, width, height) or (batch_size, seq_len, features)"
                )
        atts_vis = torch.ones(frame_features.size()[:-1], dtype=torch.long).to(
            frame_features.device
        )
        video_dict = {"video": video, "atts_vis": atts_vis}

        input_ids = output_tokenized["input_ids"]
        attention_mask = output_tokenized["attention_mask"]
        labels = input_ids.masked_fill(input_ids == -100, -100)

        if input_ids is not None:
            embedding_layer = self.lm.get_input_embeddings()
            text_features = embedding_layer(input_ids)
            input_features = torch.cat([frame_features, text_features], dim=1)
        else:
            input_features = frame_features

        video_feature_len = frame_features.size(1)
        if labels is not None:
            padding = torch.full(
                (labels.size(0), video_feature_len),
                -100,
                dtype=torch.long,
                device=labels.device,
            )
            labels = torch.cat([padding, labels], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (input_features.size(0), video_feature_len),
                    dtype=torch.long,
                    device=input_features.device,
                ),
                attention_mask,
            ],
            dim=1,
        )

        outputs = self.lm(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=input_features,
            labels=labels,
        )
        loss = outputs.loss

        return {"loss": loss}, video_dict

    @staticmethod
    def from_pretrained(
        vision_encoder_name_or_path: str,
        language_model_name_or_path: str,
        vision_projection_name_or_path: str,
        **kwargs
    ):
        vision_encoder = Blip2VisionModel.from_pretrained(
            vision_encoder_name_or_path, **kwargs
        )
        language_model = LlamaForCausalLM.from_pretrained(
            language_model_name_or_path, **kwargs
        )
        model = LlamaVideo(vision_encoder, language_model, **kwargs)
        if vision_projection_name_or_path is not None:
            model.vision_projection.load_state_dict(
                torch.load(vision_projection_name_or_path)
            )
        return model

    @torch.autocast(device_type="cuda")
    @torch.no_grad()
    def generate(
        self,
        video,
        input_tokenized,
        use_nucleus_sampling=False,
        num_beams=4,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        if len(video.size()) == 5:
            frame_features = []
            for batch_idx in range(video.size(0)):
                item = video[batch_idx].contiguous()
                encoded_item = self.vision_encoder(pixel_values=item)
                frame_features.append(encoded_item.last_hidden_state[:, 0, :])
            frame_features = torch.stack(frame_features)
            frame_features = self.vision_projection(frame_features)
        elif len(video.size()) == 3:
            frame_features = self.vision_projection(video)
        else:
            raise ValueError(
                "Invalid input shape. Must be either (batch_size, seq_len, channels, width, height) or (batch_size, seq_len, features)"
            )

        video_feature_len = frame_features.size(1)
        attention_mask = torch.ones(
            (frame_features.size(0), video_feature_len),
            dtype=torch.long,
            device=frame_features.device,
        )

        outputs = self.lm.generate(
            input_ids=None,
            inputs_embeds=frame_features,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text
