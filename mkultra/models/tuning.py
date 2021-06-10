from transformers import GPT2LMHeadModel, GPTNeoForCausalLM
from mkultra.soft_prompt import SoftPrompt
import torch
import torch.nn as nn

class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        for param in model.parameters():
            param.requires_grad = False

        model.initialize_soft_prompt()

        return model

    def initialize_soft_prompt(self, n_tokens = 20):
        self.learned_embedding = nn.parameter.Parameter(
            self.transformer.wte.weight[:n_tokens].clone().detach())

    def set_soft_prompt_embeds(self, soft_prompt_embeds):
        self.learned_embedding = nn.parameter.Parameter(soft_prompt_embeds.clone().detach())

    def set_soft_prompt(self, sp: SoftPrompt):
        self.learned_embedding = nn.parameter.Parameter(sp.get_inputs_embeds().clone().detach())

    def get_soft_params(self):
        return self.learned_embedding

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        # Drop 'past' to make things easier for us later
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)

    def _cat_learned_embedding_to_input(self, input_ids):
        #print(f"input_ids {input_ids}")

        n_tokens = self.learned_embedding.shape[-2]
        #print(f"n_tokens {n_tokens}")

        inputs_embeds = self.transformer.wte(input_ids)
        #print(f"inputs_embeds {inputs_embeds.shape}")

        # Prefix the input embeds with the learned embedding
        inputs_embeds = torch.cat([self.learned_embedding.repeat(inputs_embeds.size(0), 1, 1),
                                   inputs_embeds],
                                   dim=1)

        #print(f"inputs_embeds after cat {inputs_embeds.shape}")
        return inputs_embeds

    def _extend_labels(self, labels):
        n_tokens = self.learned_embedding.shape[-2]

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = labels.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), labels], dim=1)

    def _extend_position_ids(self, position_ids):
        # Not seen anything use this so put a breakpoint for now
        breakpoint()
        pass

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.learned_embedding.shape[-2]
        n_batches = attention_mask.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), attention_mask], dim=1)

    def forward(self, *args, **kwargs):

        if kwargs.get('input_ids') is not None:
            kwargs['inputs_embeds'] = self._cat_learned_embedding_to_input(kwargs.get('input_ids'))
            kwargs['input_ids'] = None
        else:
            pass

        if kwargs.get('labels') is not None:
            kwargs['labels'] =  self._extend_labels(kwargs.get('labels'))
        else:
            pass

        if kwargs.get('position_ids') is not None:
            print(f"position_ids.shape {kwargs['position_ids'].shape}")
            kwargs['position_ids'] = self._extend_position_ids(kwargs.get('position_ids'))
        else:
            #print("No position_ids")
            pass

        if kwargs.get('attention_mask') is not None:
            print(f"attention_mask.shape {kwargs['attention_mask'].shape}")
            kwargs['attention_mask'] = self._extend_attention_mask(kwargs.get('attention_mask'))
        else:
            #print("No attention_mask")
            pass

        if kwargs.get('token_type_ids') is not None:
            print(f"token_type_ids.shape {kwargs['token_type_ids'].shape}")
            breakpoint()
        else:
            #print("No token_type_ids")
            pass

        if kwargs.get('head_mask') is not None:
            print(f"head_mask.shape {kwargs['head_mask'].shape}")
            breakpoint()
        else:
            #print("No head_mask")
            pass

        kwargs['input_ids'] = None

        return super().forward(*args, **kwargs)

class GPT2PromptTuningLM(GPTPromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoPromptTuningLM(GPTPromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)