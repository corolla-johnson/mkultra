from transformers.pipelines import pipeline
from transformers import AutoTokenizer
from mkultra import GPT2SoftPromptLM, SoftPrompt
import torch

#===========================================#
# With pipelines and raw text (recommended) #
#===========================================#

# You'll need to instantiate one of mkultra's model classes, but the tokenizer can be as usual.
model = GPT2SoftPromptLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SoftPrompts may be loaded in one of several ways.
sp = SoftPrompt.from_file("soft_prompts/neuromancer.pt")
# sp = SoftPrompt.from_inputs_embeds(inputs_embeds)
# sp = SoftPrompt.from_learned_embedding(model.transformer.wte.learned_embedding)
# sp = SoftPrompt.from_string("The quick brown fox jumped over the lazy dog.", model, tokenizer)

# The tokenizer vocab must be updated every time new soft prompts are loaded.
tokenizer.add_special_tokens(SoftPrompt.get_special_tokens())

# Use get_tag_str() to insert the soft prompt into a string context.
prompt = sp.get_tag_str() + "\nThe sky over the port"

# The tag string conveniently contains the SP's name, part of its GUID,
# and a series of '@'s which represent individual soft tokens.
print(prompt)
# <neuromancer-11084ee4-@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@>
# The sky over the port

# These tokens help you budget your context.
print(f"Length of soft prompt: {len(tokenizer.encode(sp.get_tag_str()))}")
print(f"Length of full prompt: {len(tokenizer.encode(prompt))}")
# Length of soft prompt: 20
# Length of full prompt: 26

# Generation is as usual.
print(generator(prompt))

#================================#
# With generate() and id tensors #
#================================#

input_ids_a = sp.get_input_ids()
input_ids_b = tokenizer("\nThe sky over the port", return_tensors="pt").input_ids
input_ids = torch.cat((input_ids_a, input_ids_b),0)

print(tokenizer.decode(model.generate(input_ids)[0]))

#======================================#
# With forward() and embedding tensors #
#======================================#

inputs_embeds_a = sp.get_inputs_embeds()
inputs_embeds_b = model.transformer.wte(tokenizer("\nThe sky over the port", return_tensors="pt").input_ids)
inputs_embeds = torch.cat((inputs_embeds_a, inputs_embeds_b),1)

output = model(inputs_embeds=inputs_embeds)