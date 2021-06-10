from transformers import GPT2TokenizerFast
from mkultra.tuning import GPT2PromptTuningLM
from mkultra.soft_prompt import SoftPrompt
import pytest
from mkultra.inference import GPT2SoftPromptLM
from mkultra.tokenizers import GPT2SPTokenizerFast

inf_model = GPT2SoftPromptLM.from_pretrained("gpt2")
inf_tokenizer = GPT2SPTokenizerFast.from_pretrained("gpt2")

tun_model = GPT2PromptTuningLM.from_pretrained("gpt2")
tun_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

@pytest.fixture(scope="session", autouse=True)
def inference_resources(request):
    return (inf_model, inf_tokenizer)

@pytest.fixture(scope="session", autouse=True)
def tuning_resources(request):
    tun_model.initialize_soft_prompt()
    return (tun_model, tun_tokenizer)