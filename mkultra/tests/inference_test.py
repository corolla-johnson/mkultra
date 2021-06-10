from transformers.pipelines import pipeline
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

def test_pipeline_inference(inference_resources):
    model, tokenizer = inference_resources
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    sp = SoftPrompt.from_string("The quick brown fox", model=model, tokenizer=tokenizer)
    prompt = sp + " jumps over the lazy dog"

    prompt_len = len(tokenizer.encode(prompt))

    output = generator( prompt,
                        do_sample=False,
                        min_length=prompt_len+2,
                        max_length=prompt_len+2,
                        use_cache=True,
                        return_full_text=True)

    output_str = output[0]['generated_text']

    # Assert no more special tokens got generated
    assert output_str.count(sp._unique_token_str()) == 1
    assert output_str.count(SoftPrompt.GENERIC_SOFT_TOKEN_STR) == len(sp) - 1

def test_generate_inference(inference_resources):
    model, tokenizer = inference_resources

    sp = SoftPrompt.from_string("The quick brown fox", model=model, tokenizer=tokenizer)
    prompt = sp + " jumps over the lazy dog"

    prompt = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt.shape[-1]

    output = model.generate(prompt,
                            do_sample=False,
                            min_length=prompt_len+2,
                            max_length=prompt_len+2,
                            use_cache=True,
                            return_full_text=False)

    output_str = tokenizer.decode(output[0])

    # Assert no more special tokens got generated
    assert output_str.count(sp._unique_token_str()) == 1
    assert output_str.count(SoftPrompt.GENERIC_SOFT_TOKEN_STR) == len(sp) - 1

def test_forward_inference(inference_resources):
    model, tokenizer = inference_resources

    sp = SoftPrompt.from_string("The quick brown fox", model=model, tokenizer=tokenizer)
    prompt = sp + " jumps over the lazy dog"

    prompt = tokenizer(prompt, return_tensors="pt").input_ids

    model(prompt)

    # Don't bother sampling, without logit bias this *will* output special tokens
