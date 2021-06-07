from mkultra.models.inference import GPT2SoftPromptLM
from mkultra.tokenizers import GPT2SPTokenizerFast
from mkultra.soft_prompt import SoftPrompt

def test_tokenizer_doesnt_break_special_tokens():
    # Arrange
    model = GPT2SoftPromptLM.from_pretrained("gpt2")
    tokenizer = GPT2SPTokenizerFast.from_pretrained("gpt2")

    test_str = " a b c d e f g"
    sp = SoftPrompt.from_string(test_str, model=model, tokenizer=tokenizer)
    exp_length = len(sp)

    # Act
    length = len(tokenizer.encode(sp.get_tag_str()))

    # Assert
    assert length == exp_length
