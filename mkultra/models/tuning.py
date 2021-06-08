from mkultra.soft_embedding import SoftEmbedding
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM
import torch

class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        s_wte = SoftEmbedding(model.get_input_embeddings(),
                              n_tokens=10,
                              initialize_from_vocab=True).to("cuda")
        model.set_input_embeddings(s_wte)

class GPT2PromptTuningLM(GPTPromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoPromptTuningLM(GPTPromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)