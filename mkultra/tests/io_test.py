import os
from mkultra.soft_prompt import SoftPrompt
import torch
import json

def test_json(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    sp_a = SoftPrompt.from_string(" a b c d e f g", model=model, tokenizer=tokenizer)

    # Act
    sp_str = sp_a.to_json()
    sp_b = SoftPrompt.from_json(sp_str)

    # Assert
    assert torch.equal(sp_a._tensor, sp_b._tensor)
    assert sp_a._metadata['name'] == sp_b._metadata['name']
    assert sp_a._metadata['description'] == sp_b._metadata['description']

def test_json_embedding(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    sp_a = SoftPrompt.from_string(" a b c d e f g", model=model, tokenizer=tokenizer)

    # Act
    with open("TEST.json", mode='w') as file:
        sp_str = sp_a.to_json()
        sp_dict = json.dump(
            { 'additional_data' : "The quick brown fox jumps over the lazy dog",
              'sp' : sp_str }, file)

    with open("TEST.json", mode='r') as file:
        sp_dict = json.load(file)

    sp_b = SoftPrompt.from_json(sp_dict['sp'])

    # Assert
    assert torch.equal(sp_a._tensor, sp_b._tensor)
    assert sp_a._metadata['name'] == sp_b._metadata['name']
    assert sp_a._metadata['description'] == sp_b._metadata['description']

    # Teardown
    os.remove("TEST.json")

def test_file_io(inference_resources):
    model, tokenizer = inference_resources

    # Arrange
    sp_a = SoftPrompt.from_string(" a b c d e f g", model=model, tokenizer=tokenizer)

    # Act
    sp_a.to_file("TEST.json")
    sp_b = SoftPrompt.from_file("TEST.json")

    # Assert
    assert torch.equal(sp_a._tensor, sp_b._tensor)
    assert sp_a._metadata['name'] == sp_b._metadata['name']
    assert sp_a._metadata['description'] == sp_b._metadata['description']

    # Teardown
    os.remove("TEST.json")

def test_file_input_only(inference_resources):
    model, tokenizer = inference_resources

    # How to recreate the test file:
    #sp = SoftPrompt.from_string("TEST", model, tokenizer)
    #sp.to_file("sample_sps/testing/iotest.json")

    # Arrange
    exp_string = "TEST"
    exp_tensor = model.get_input_embeddings()(tokenizer(exp_string, return_tensors="pt").input_ids)

    # Act
    sp_a = SoftPrompt.from_file("sample_sps/testing/iotest.json")

    # Assert
    assert torch.equal(sp_a._tensor, exp_tensor)