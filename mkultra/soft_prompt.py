import json
import uuid
import datetime

class SoftPrompt:
    """
    A soft prompt.

    After loading any number of soft prompts, make sure to prime your tokenizer with
    tokenizer.add_special_tokens(SoftPrompt.get_special_tokens()).

    Attributes:
        _tensor: The underlying tensor.
        _metadata: A metadata dictionary. Keys are:
                   'name', 'uuid', 'description', 'date'

    Class attributes:
        _loaded_sps: A static dictionary where
                 key - a special input_id
                 value - a SoftPrompt corresponding to that input_id
    """
    GENERIC_SOFT_TOKEN_STR = "<@>"
    _loaded_soft_prompts = dict()
    _tokenizer = None
    _model = None

    def __init__(self, tensor, metadata=None):
        self._tensor = tensor
        self._metadata = metadata

    def __len__(self):
        return self._tensor.shape[1]

    def __str__(self):
        return (f"{self._metadata['name']} ({self._metadata['date']})\n"
                f"Length: {len(self)}\n"
                f"UUID:   {self._metadata['uuid']}\n"
                f"Description:\n"
                f"{self._metadata['description']}")

    def _check_integrity(self):
        if self._tensor is None:
            raise AttributeError("Malformed SoftPrompt has no _tensor attribute")

        # Generate metadata if missing
        if self._metadata is None:
            self._metadata = dict()

        if self._metadata.get('name') is None:
            self._metadata['name'] = "Untitled"

        if self._metadata.get('uuid') is None:
            self._metadata['uuid'] = str(uuid.uuid4())

        if self._metadata.get('description') is None:
            self._metadata['description'] = "No description given."

        if self._metadata.get('date') is None:
            self._metadata['date'] = datetime.datetime.now()

    def _unique_token_str(self):
        return f"<{self._metadata['name']}-{self._metadata['uuid'][:8]}-@>"

    @staticmethod
    def _add_to_loaded_sps(sp):
        SoftPrompt._tokenizer.add_tokens(sp._unique_token_str())
        SoftPrompt._model.resize_token_embeddings(len(SoftPrompt._tokenizer))

        input_id = SoftPrompt._tokenizer(sp._unique_token_str(), return_tensors="pt").input_ids[0,0].item()

        print(f"added input_id {input_id}")

        SoftPrompt._loaded_soft_prompts[input_id] = sp

    @staticmethod
    def setup_tokenizer(tokenizer, model):
        """Sets up a tokenizer for soft prompts.

        Needs to called before using a new tokenizer.
        Using multiple tokenizers at the same time is not supported and may cause unexpected behaviour.
        """
        SoftPrompt._tokenizer = tokenizer
        SoftPrompt._model = model

        tokenizer.add_tokens(SoftPrompt.get_special_tokens())
        model.resize_token_embeddings(len(tokenizer))

        # Use tokenizer to rebuild _loaded_sps
        new_loaded_soft_prompts = dict()

        for sp in SoftPrompt._loaded_soft_prompts.values():
            input_id = tokenizer(sp._unique_token_str(), return_tensors="pt").input_ids[0,0].item()
            new_loaded_soft_prompts[input_id] = sp

        SoftPrompt._loaded_soft_prompts = new_loaded_soft_prompts

        return tokenizer

    @staticmethod
    def get_special_tokens():
        """Returns the list of special tokens for all loaded soft prompts.
        """
        special_tokens = list()
        special_tokens.append(SoftPrompt.GENERIC_SOFT_TOKEN_STR)

        for sp in SoftPrompt._loaded_soft_prompts.values():
            special_tokens.append(sp._unique_token_str())

        return special_tokens

    @staticmethod
    def from_file(path):
        """Loads a soft prompt from a file.
        """
        with open(path) as file:
            sp = json.load(file)
            sp._check_integrity()

            # Check if this soft prompt's uuid already exists
            old_sp = [x for x in SoftPrompt._loaded_soft_prompts.values()
                      if x._metadata['uuid'] == x._metadata['uuid']]

            if old_sp.count != 0:
                sp._metadata['uuid'] = str(uuid.uuid4())

            SoftPrompt._add_to_loaded_sps(sp)
            return sp

    @staticmethod
    def from_input_id(input_id):
        """Gets the already-loaded soft prompt corresponding to a special input_id.

        Returns:
            a SoftPrompt if a corresponding one exists,
            otherwise None
        """
        return SoftPrompt._loaded_soft_prompts.get(input_id)

    @staticmethod
    def from_inputs_embeds(inputs_embeds, metadata=None):
        """Creates a soft prompt from an embedding tensor.
        """
        sp = SoftPrompt(tensor=inputs_embeds, metadata=metadata)
        sp._check_integrity()
        SoftPrompt._add_to_loaded_sps(sp)
        return sp

    @staticmethod
    def from_tuning_model(model, metadata=None):
        """Extracts a soft prompt from a PromptTuningModel.
        """
        raise NotImplementedError()

    @staticmethod
    def from_string(string, model, metadata=None):
        """Creates a soft prompt by tokenizing and embedding a string.

        This is useful for testing as it is repeatable.
        You can also use this method to set a starting point for training.
        """
        tokens = SoftPrompt._tokenizer(string, return_tensors="pt").input_ids
        inputs_embeds = model.get_input_embeddings()(tokens)
        return SoftPrompt.from_inputs_embeds(inputs_embeds=inputs_embeds, metadata=metadata)

    def get_tag_str(self):
        """Returns a string used to mark a location for this soft prompt.

        This will consist of unique special token containing the prompt's name and UUID,
        followed by a number of '<@>'s that represent individual soft tokens.
        """
        tag_str = self._unique_token_str()

        for _ in range(len(self)-1):
            tag_str += SoftPrompt.GENERIC_SOFT_TOKEN_STR

        return tag_str

    def get_inputs_embeds(self):
        """Returns the embedding tensor of this soft prompt.
        """
        return self._tensor

