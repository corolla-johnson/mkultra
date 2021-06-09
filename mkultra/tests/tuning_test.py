from mkultra.soft_prompt import SoftPrompt
import torch

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
    output = model._cat_learned_embedding_to_input(input_ids, labels)

    # Assert
    assert torch.equal(output[0], exp_inputs_embeds)
    assert torch.equal(output[1], exp_labels)