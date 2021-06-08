from mkultra.models.inference import GPT2SoftPromptLM
from mkultra.tokenizers import GPT2SPTokenizerFast
from mkultra.soft_prompt import SoftPrompt
import torch

def test_replace_special_tokens():
    # Arrange
    model = GPT2SoftPromptLM.from_pretrained("gpt2")
    tokenizer = GPT2SPTokenizerFast.from_pretrained("gpt2")

    a = " 1 2 3"
    b = " 4 5 6"
    c = " 7 8 9"
    a_b_c = a + b + c

    sp = SoftPrompt.from_string(b,model=model, tokenizer=tokenizer)
    input_ids = tokenizer(a + sp + c, return_tensors="pt").input_ids
    exp_inputs_embeds = model.get_input_embeddings()(tokenizer(a_b_c, return_tensors="pt").input_ids)

    # Act
    inputs_embeds = model.replace_special_tokens(input_ids)

    # Assert
    assert torch.equal(inputs_embeds, exp_inputs_embeds)

def test_pipeline_all_ok():

    #Assert no special tokens in output
    pass

def test_cuda_inference_ok():
    pass