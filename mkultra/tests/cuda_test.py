from transformers.pipelines import pipeline
from mkultra.soft_prompt import SoftPrompt
from mkultra.inference import GPT2SoftPromptLM
from mkultra.tuning import GPT2PromptTuningLM
import torch

def test_cuda_inference(inference_resources):
    model = GPT2SoftPromptLM.from_pretrained("gpt2").to("cuda")
    _, tokenizer = inference_resources

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

def test_cuda_inference_with_tuning_model(tuning_resources):
    model = GPT2PromptTuningLM.from_pretrained("gpt2").to("cuda")
    _, tokenizer = tuning_resources

    prompt = "The quick brown fox jumps over the lazy dog"

    prompt = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt.shape[-1]

    output = model.generate(input_ids=prompt,
                            do_sample=False,
                            min_length=prompt_len+2,
                            max_length=prompt_len+2,
                            use_cache=True,
                            return_full_text=False)

    output_str = tokenizer.decode(output[0])

    # Just make sure nothing breaks
