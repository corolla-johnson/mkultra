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
        inputs_embeds = self.transformer.wte(input_ids)

        inputs_embeds = torch.cat([self.learned_embedding.repeat(inputs_embeds.size(0), 1, 1),
                                   inputs_embeds],
                                   dim=1)

        return inputs_embeds

    def _extend_labels(self, labels):
        n_tokens = self.learned_embedding.shape[-2]

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = labels.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), labels], dim=1)

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.learned_embedding.shape[-2]
        n_batches = attention_mask.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), attention_mask], dim=1)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

class GPT2PromptTuningLM(GPTPromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoPromptTuningLM(GPTPromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)