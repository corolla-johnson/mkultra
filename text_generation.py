from transformers.pipelines import pipeline
from mkultra.models.inference import GPT2SoftPromptLM
from mkultra.tokenizers import GPT2SPTokenizerFast
from mkultra.soft_prompt import SoftPrompt

# You'll need to instantiate mkultra's model and tokenizer classes.
model = GPT2SoftPromptLM.from_pretrained("gpt2")
tokenizer = GPT2SPTokenizerFast.from_pretrained("gpt2")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SoftPrompts may be loaded in one of several ways.
# sp = SoftPrompt.from_file("soft_prompts/neuromancer.pt")
# sp = SoftPrompt.from_inputs_embeds(inputs_embeds)
# sp = SoftPrompt.from_learned_embedding(model.transformer.wte.learned_embedding)
# We will instantiate an SP from a string for testing. This should behave identically to the text.
sp = SoftPrompt.from_string("With the court firmly balkanized into three distinct factions, Princess Charlotte had her work cut out for her.",
                            model=model, tokenizer=tokenizer)

# Information about an sp can be printed with
print(sp)

# Use 'get_tag_str()' to get a tag string that you can add to your context.
prompt = sp.get_tag_str() + " She"

# The addition operator also works:
# prompt = sp + " She"

# The tag string conveniently contains the SP's name, part of its GUID,
# and a series of '@'s which represent individual soft tokens.
print(prompt)
# <FromString-11084ee4-@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@><@> She

# These tokens help you budget your context.
prompt_len = len(tokenizer.encode(prompt))
print(f"Length of soft prompt: {len(tokenizer.encode(sp.get_tag_str()))}")
print(f"Length of full prompt: {prompt_len}")
# Length of soft prompt: 22
# Length of full prompt: 23

# Generation is as usual.
output = generator( prompt,
                    do_sample=True,
                    min_length=prompt_len+100,
                    max_length=prompt_len+100,
                    repetition_penalty=1.7,
                    top_p=0.8,
                    temperature=0.7,
                    use_cache=True,
                    return_full_text=True)

print(output)

# WARNING: If you decide to use model.forward(),
# be aware that the special token logits will be very high.
# This is accounted for when using model.generate() or pipelines,
# but you will need to exclude special tokens when sampling from logits.
