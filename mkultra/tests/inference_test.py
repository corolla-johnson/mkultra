from mkultra.soft_prompt import SoftPrompt
import torch

def test_replace_special_tokens(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
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

def test_pipeline_inference():
    #Assert no special tokens in output
    pass

def test_generate_inference():
    #Assert no special tokens in output
    pass

def test_forward_inference():
    #Assert no special tokens in output
    pass
