from mkultra.soft_prompt import SoftPrompt

def test_tokenizer_doesnt_break_special_tokens(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    test_str = " a b c d e f g"
    sp = SoftPrompt.from_string(test_str, model=model, tokenizer=tokenizer)
    exp_length = len(sp)

    # Act
    length = len(tokenizer.encode(sp.get_tag_str()))

    # Assert
    assert length == exp_length
    assert test_str != sp.get_tag_str()
