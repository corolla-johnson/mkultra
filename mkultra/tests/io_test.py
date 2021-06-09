import os
from mkultra.soft_prompt import SoftPrompt
import torch

def test_serialization(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    sp_a = SoftPrompt.from_string(" a b c d e f g", model=model, tokenizer=tokenizer)

    # Act
    sp_str = sp_a.serialize()
    sp_b = SoftPrompt.from_serialized(sp_str)

    # Assert
    assert torch.equal(sp_a._tensor, sp_b._tensor)
    assert sp_a._metadata['name'] == sp_b._metadata['name']
    assert sp_a._metadata['description'] == sp_b._metadata['description']

def test_file_io(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    sp_a = SoftPrompt.from_string(" a b c d e f g", model=model, tokenizer=tokenizer)

    # Act
    sp_a.to_file("TEST.pt")
    sp_b = SoftPrompt.from_file("TEST.pt")

    # Assert
    assert torch.equal(sp_a._tensor, sp_b._tensor)
    assert sp_a._metadata['name'] == sp_b._metadata['name']
    assert sp_a._metadata['description'] == sp_b._metadata['description']

    # Teardown
    os.remove("TEST.pt")

def test_file_input_only(inference_resources):
    model, tokenizer = inference_resources

    # How to recreate the test file:
    # sp = SoftPrompt.from_string("TEST", model, tokenizer)
    # sp.to_file("sample_sps/testing/iotest.pt")
    # model should be on CPU

    # Arrange
    exp_string = "TEST"
    exp_tensor = model.get_input_embeddings()(tokenizer(exp_string, return_tensors="pt").input_ids)

    # Act
    sp_a = SoftPrompt.from_file("sample_sps/testing/iotest.pt")

    # Assert
    assert torch.equal(sp_a._tensor, exp_tensor)