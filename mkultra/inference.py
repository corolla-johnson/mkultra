from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, TextGenerationPipeline, GPTJForCausalLM
from mkultra.soft_prompt import SoftPrompt
import torch

EXTRA_ALLOWED_MODELS = [
    "GPT2SoftPromptLM",
    "GPTNeoSoftPromptLM"
    "GPTJSoftPromptLM"
    ]

for model in EXTRA_ALLOWED_MODELS:
    if model not in TextGenerationPipeline.ALLOWED_MODELS:
        TextGenerationPipeline.ALLOWED_MODELS.append(model)

class GPTSoftPromptMixin:
    def replace_special_tokens(self, input_ids):
        # Embed everything normally first
        inputs_embeds = self.transformer.wte(input_ids.to(self.device))

        n_batches = input_ids.shape[0]
        n_tokens = input_ids.shape[-1]

        for b in range(n_batches):
            # Replace special tokens with soft prompts
            for t in range(n_tokens):
                # Check each id for a special token
                input_id = input_ids[b,t].item()
                sp = SoftPrompt.from_input_id(input_id)
                # If we find one, replace special token and padding with soft prompt
                if sp:
                    replacement = sp.get_inputs_embeds().to(self.device).clone().unsqueeze(0)
                    inputs_embeds[b,t:t+len(sp),:] = replacement[0,:,:]

        return inputs_embeds

    def forward(self, *args, **kwargs):
        # Alow setting input_ids as positional arg 0
        if kwargs.get('input_ids') is None:
            kwargs['input_ids'] = args[0]

        input_ids = kwargs.get('input_ids').to(self.device)

        if input_ids is None:
            # User is using inputs_embeds, nothing more we can do
            return super().forward(*args, **kwargs)

        kwargs['input_ids'] = None
        kwargs['inputs_embeds'] = self.replace_special_tokens(input_ids)

        args = ()

        return super().forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Alow setting input_ids as positional arg 0
        if kwargs.get('input_ids') is None:
            kwargs['input_ids'] = args[0]

        # This fixes CUDA for some reason
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)

        # Ban all special logits from output
        bad_words_ids = kwargs.get('bad_words_ids', list())

        for id in SoftPrompt.get_special_token_ids():
            bad_words_ids.append([id])
        kwargs['bad_words_ids'] = bad_words_ids

        args = ()

        return super().generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        SoftPrompt._register_model(model)
        return model

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(input_ids, past, *args, **kwargs)

class GPT2SoftPromptLM(GPTSoftPromptMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoSoftPromptLM(GPTSoftPromptMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)

class GPTJSoftPromptLM(GPTSoftPromptMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
