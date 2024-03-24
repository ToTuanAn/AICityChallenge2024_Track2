from typing import List, Optional, Tuple, Union 

import torch
from torch import nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         GemmaConfig, GemmaModel, GemmaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import .

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(GemmaConfig):
    model_type = "llava"

class LlavaGemmaModel(GemmaModel, LlavaMetaModel):
    config_class = LlavaConfig

    def __init__(self, config: GemmaConfig):
        super(LlavaGemmaModel, self).__init__(config)


class LlavaGemmaForCausalLM(GemmaForCausalLM, LlavaMetaForCausalLM):
    def __init__(self, config):
        super(GemmaForCausalLM, self).__init__(config)
        self.model = LlavaGemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_state, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            images,
            image_sizes,
            return_dict,
            cache_position
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError('"input_embeds" is not supported')

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                
            )   
        pass
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
                                      ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

AutoConfig.register("llava_gemma", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaGemmaForCausalLM)
