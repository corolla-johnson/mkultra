from mkultra.models.inference import GPT2SoftPromptLM
from mkultra.tokenizers import GPT2SPTokenizerFast
from mkultra.soft_prompt import SoftPrompt

def test_instantiate_multiple_objects():
    # Act
    model_a = GPT2SoftPromptLM.from_pretrained("gpt2")
    tokenizer_a = GPT2SPTokenizerFast.from_pretrained("gpt2")

    sp_a = SoftPrompt.from_string("TEST",model=model_a, tokenizer=tokenizer_a)

    model_b = GPT2SoftPromptLM.from_pretrained("gpt2")
    tokenizer_b = GPT2SPTokenizerFast.from_pretrained("gpt2")

    sp_b = SoftPrompt.from_string("TEST",model=model_b, tokenizer=tokenizer_b)

    # Assert
    assert model_a in SoftPrompt._models
    assert model_b in SoftPrompt._models
    assert tokenizer_a in SoftPrompt._tokenizers
    assert tokenizer_b in SoftPrompt._tokenizers
    assert sp_a in SoftPrompt._soft_prompts
    assert sp_b in SoftPrompt._soft_prompts

    # Teardown
    del model_a
    del model_b
    del tokenizer_a
    del tokenizer_b
