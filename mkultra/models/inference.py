from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, TextGenerationPipeline
from mkultra.soft_prompt import SoftPrompt
import torch

EXTRA_ALLOWED_MODELS = [
    "GPT2SoftPromptLM",
    "GPTNeoSoftPromptLM"
    ]

for model in EXTRA_ALLOWED_MODELS:
    if model not in TextGenerationPipeline.ALLOWED_MODELS:
        TextGenerationPipeline.ALLOWED_MODELS.append(model)

class GPTSoftPromptMixin:
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # Embed everything normally first
        inputs_embeds = self.transformer.wte(input_ids)

        # Replace special tokens with soft prompts
        for i in range(input_ids.shape[-1]):
            # Check each id for a special token
            input_id = input_ids[0,i].item()
            sp = SoftPrompt.from_input_id(input_id)
            # If we find one, replace special token and padding with soft prompt
            if sp:
                inputs_embeds[0,i:i+len(sp),:] = sp.get_inputs_embeds().to(self.device)[0,:,:]

        # Drop input_ids, pass other arguments
        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "use_cache": kwargs.get("use_cache"),
            # Don't support these arguments for now
            #"position_ids": inputs['position_ids'],
            #"attention_mask": inputs['attention_mask'],
            #"token_type_ids": inputs['token_type_ids'],
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Ban all special logits from output
        bad_words_ids = kwargs.get('bad_words_ids', list())

        for id in SoftPrompt.get_special_token_ids():
            bad_words_ids.append([id])
        kwargs['bad_words_ids'] = bad_words_ids

        return super().generate(*args, **kwargs)


class GPT2SoftPromptLM(GPTSoftPromptMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        SoftPrompt._register_model(self)

class GPTNeoSoftPromptLM(GPTSoftPromptMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        SoftPrompt._register_model(self)