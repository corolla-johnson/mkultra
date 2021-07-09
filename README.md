# mkultra
mkultra is a prompt tuning toolkit for GPT-2 and GPT-Neo.

Prompt tuning injects a string of 20-100 special tokens into the context in order to influence text generation. These tokens are trained on a corpus much like a finetune, but take up a fraction of the space. The Neuromancer example is only 401kb for 100 tokens.

Read the original paper: https://arxiv.org/abs/2104.08691


## Text Generation
```
model = GPT2SoftPromptLM.from_pretrained("gpt2")
tokenizer = GPT2SPTokenizerFast.from_pretrained("gpt2")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

sp = SoftPrompt.from_file("sample_sps/finetune/neuromancer_gpt2.json")
prompt = sp + "The sky over the port"
output = generator(prompt)
```
SoftPrompts can be concatenated at any point into your context as if they were strings. When the context is printed, SoftPrompts show up as human-readable tags for debugging. They also tokenize to the underlying number of tokens for easy budgeting.

See the [text generation notebook](text_generation.ipynb) for pointers on adding mkultra to your generator.


## Training

For finetune-like soft prompts, the [finetune notebook](https://colab.research.google.com/github/corolla-johnson/mkultra/blob/master/tuning_finetune.ipynb) demonstrates training on a corpus.

For AI text adventures or writing, the [World Info notebook](https://colab.research.google.com/github/corolla-johnson/mkultra/blob/master/tuning_world_info.ipynb) notebook demonstrates tuning a soft prompt to describe a character or setting. This is highly experimental.

## Limitations (for now)

- The Huggingface Trainer class should work as long as you set params=[model.get_soft_params()] on the optimizer, but it will still save full model checkpoints.
- mkultra syncs a set of special tokens between its tokenizers the scenes. Adding your own tokens may result in unexpected behaviour.
