from mkultra.soft_prompt import SoftPrompt
import torch

def test_tuning_forward(tuning_resources):
    model, tokenizer = tuning_resources
    model.initialize_soft_prompt(n_tokens=20)

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    labels = input_ids.clone().detach()

    model(input_ids=input_ids, labels=labels)

def test_cat_learned_embedding(tuning_resources):
    model, tokenizer = tuning_resources

    # Arrange
    a = " 1 2 3"
    b = " 4 5 6"
    a_b = a + b

    sp = SoftPrompt.from_string(a,model=model, tokenizer=tokenizer)
    model.set_soft_prompt(sp)

    input_ids = tokenizer(b, return_tensors="pt").input_ids
    labels = input_ids.clone().detach()
    exp_inputs_embeds = model.get_input_embeddings()(tokenizer(a_b, return_tensors="pt").input_ids)
    exp_labels = torch.cat([torch.full((1,len(sp)), -100), labels], dim=1)

    # Act
    inputs_embeds = model._cat_learned_embedding_to_input(input_ids)
    labels = model._extend_labels(labels)

    # Assert
    assert torch.equal(inputs_embeds, exp_inputs_embeds)
    assert torch.equal(labels, exp_labels)
