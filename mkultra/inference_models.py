from transformers import GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from mkultra import SoftPrompt

class GPTSoftPromptMixin:
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(self, input_ids, past=past, **kwargs)

        # Embed everything normally first
        inputs_embeds = self.transformer.wte(inputs['input_ids'])

        # Replace special tokens with soft prompts
        for i in range(input_ids.shape[-1]):
            # Check each id for a special token
            sp = SoftPrompt.from_special_token(input_ids[:,i])
            # If we find one, replace special token and padding with soft prompt
            if sp:
                inputs_embeds[0,i:i+len(sp),:] = sp.to(self.device).get_input_embeds()[0,:,:]

        # Drop input_ids, pass other arguments
        return {
            "inputs_embeds": inputs_embeds,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": inputs['position_ids'],
            "attention_mask": inputs['attention_mask'],
            "token_type_ids": inputs['token_type_ids'],
        }

class GPT2SoftPromptLM(GPTSoftPromptMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoSoftPromptLM(GPTSoftPromptMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)