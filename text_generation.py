from transformers.pipelines import pipeline
from transformers import GPT2TokenizerFast
from mkultra.inference_models import GPT2SoftPromptLM
from mkultra.soft_prompt import SoftPrompt
import torch

# You'll need to instantiate one of mkultra's model classes.
model = GPT2SoftPromptLM.from_pretrained("gpt2")

# SoftPrompt.setup_tokenizer() should be called before using a new tokenizer.
tokenizer = SoftPrompt.setup_tokenizer(GPT2TokenizerFast.from_pretrained("gpt2"), model)

# Set up your generator as usual.
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SoftPrompts may be loaded in one of several ways.
# sp = SoftPrompt.from_file("soft_prompts/neuromancer.pt")
# sp = SoftPrompt.from_inputs_embeds(inputs_embeds)
# sp = SoftPrompt.from_learned_embedding(model.transformer.wte.learned_embedding)
sp = SoftPrompt.from_string("With the court firmly balkanized into three distinct factions, Princess Charlotte had her work cut out for her.", model)

print(sp.get_inputs_embeds())

# Use get_tag_str() to insert the soft prompt into a string context.
prompt = sp.get_tag_str() + " Her"

pl = len(tokenizer.encode(prompt))

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

print(f"special input_ids:{SoftPrompt._loaded_soft_prompts}")

bad_words_ids = list()
for token in SoftPrompt.get_special_tokens():
    bad_words_ids.append(tokenizer.encode(token))

print(bad_words_ids)

# Generation is as usual.
output = generator( prompt,
                    do_sample=True,
                    min_length=pl+100,
                    max_length=pl+100,
                    repetition_penalty=1.7,
                    top_p=0.8,
                    temperature=0.7,
                    bad_words_ids=bad_words_ids,
                    use_cache=True,
                    return_full_text=True)

print(output)

# WARNING: If you decide to use model.forward(),
# be aware that the special token logits will be very high.
# This is accounted for when using model.generate() or pipelines,
# but you will need to exclude special tokens when sampling from logits.
