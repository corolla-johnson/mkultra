from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, TextGenerationPipeline
from mkultra.soft_prompt import SoftPrompt

EXTRA_ALLOWED_MODELS = [
    "GPT2SoftPromptLM",
    "GPTNeoSoftPromptLM"
    ]

for model in EXTRA_ALLOWED_MODELS:
    if model not in TextGenerationPipeline.ALLOWED_MODELS:
        TextGenerationPipeline.ALLOWED_MODELS.append(model)

class GPTSoftPromptMixin:
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):

        print(input_ids)

        # Embed everything normally first
        inputs_embeds = self.transformer.wte(input_ids)

        # Replace special tokens with soft prompts
        for i in range(input_ids.shape[-1]):
            # Check each id for a special token
            input_id = input_ids[0,i].item()
            print(f"Looking for input id {input_id}")
            sp = SoftPrompt.from_input_id(input_id)
            # If we find one, replace special token and padding with soft prompt
            if sp:
                print("replacing")
                print(inputs_embeds)
                inputs_embeds[0,i:i+len(sp),:] = sp.get_inputs_embeds().to(self.device)[0,:,:]
                print(inputs_embeds)

        print(input_ids.shape)
        print(inputs_embeds.shape)

        # Drop input_ids, pass other arguments
        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "use_cache": kwargs.get("use_cache"),
            # Don't support custom attention masks for now
            #"position_ids": inputs['position_ids'],
            #"attention_mask": inputs['attention_mask'],
            #"token_type_ids": inputs['token_type_ids'],
        }

class GPT2SoftPromptLM(GPTSoftPromptMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoSoftPromptLM(GPTSoftPromptMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)