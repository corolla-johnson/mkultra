from transformers.pipelines import pipeline
from transformers import GPT2TokenizerFast
from mkultra.inference_models import GPT2SoftPromptLM
from mkultra.soft_prompt import SoftPrompt
import torch

# You'll need to instantiate one of mkultra's model classes.
model = GPT2SoftPromptLM.from_pretrained("gpt2")

# SoftPrompt.setup_tokenizer() should be called before using a new tokenizer.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
SoftPrompt.setup_tokenizer(tokenizer)

# Set up your generator as usual.
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SoftPrompts may be loaded in one of several ways.
sp = SoftPrompt.from_file("soft_prompts/neuromancer.pt")
# sp = SoftPrompt.from_inputs_embeds(inputs_embeds)
# sp = SoftPrompt.from_learned_embedding(model.transformer.wte.learned_embedding)
# sp = SoftPrompt.from_string("The quick brown fox jumped over the lazy dog.", model, tokenizer)

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