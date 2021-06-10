import json
import uuid
import datetime
import torch
from typing import Dict, Any, Union
import pickle
import base64

class SoftPrompt():
    """
    A soft prompt.

    After loading any number of soft prompts, make sure to prime your tokenizer with
    tokenizer.add_special_tokens(SoftPrompt.get_special_tokens()).

    Attributes:
        _tensor: The underlying tensor.
        _metadata: A metadata dictionary. Keys are:
                   'name', 'uuid', 'description', 'epoch'

    Class attributes:
        _loaded_sps: A static dictionary where
                 key - a special input_id
                 value - a SoftPrompt corresponding to that input_id
    """
    # Token that represents a generic soft prompt.
    GENERIC_SOFT_TOKEN_STR = "<@>"

    # LUT that maps int input ids to SoftPrompts.
    _id_lut = dict()

    # List of loaded SoftPrompts.
    _soft_prompts = list()

    # List of compatible tokenizers.
    _tokenizers = list()

    # List of compatible models.
    _models = list()

    def __init__(self, tensor: torch.Tensor, metadata: Dict[str, Any]=None):
        self._tensor = tensor

        #if len(self._tensor.shape) == 2:
        #    self._tensor.unsqueeze_(0)

        self._metadata = metadata

    def __len__(self):
        return self._tensor.shape[-2]

    def __str__(self):
        return (f"{self._metadata['name']} ({datetime.datetime.fromtimestamp(self._metadata['epoch'])})\n"
                f"Length: {len(self)}\n"
                f"UUID:   {self._metadata['uuid']}\n"
                f"Description:\n"
                f"{self._metadata['description']}")

    def __add__(self, other: str):
        return self.get_tag_str() + other

    def __radd__(self, other: str):
        return other + self.get_tag_str()

    def _unique_token_str(self):
        return f"<{self._metadata['name']}-{self._metadata['uuid'][:8]}-@>"

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

        if self._metadata.get('epoch') is None:
            self._metadata['epoch'] = datetime.datetime.now().timestamp()

    @staticmethod
    def _register_soft_prompt(sp: 'SoftPrompt'):
        SoftPrompt._soft_prompts.append(sp)

        # We can wait until a suitable tokenizer gets created
        if len(SoftPrompt._tokenizers) > 0:
            SoftPrompt._add_tokens_to_tokenizers()
            SoftPrompt._resize_model_embeddings()
            SoftPrompt._refresh_id_lut()

    @staticmethod
    def _refresh_id_lut():
        # This clearly won't work for mulitple tokenizers with unique vocabularies,
        # but it should cover standard use cases (incl. extra vocab on one tokenizer)
        for sp in SoftPrompt._soft_prompts:
            input_id = SoftPrompt._tokenizers[0].encode(sp._unique_token_str())[0]
            SoftPrompt._id_lut[input_id] = sp

    @staticmethod
    def _add_tokens_to_tokenizers():
        for tokenizer in SoftPrompt._tokenizers:
            tokenizer.add_tokens(SoftPrompt.get_special_token_strs())

    @staticmethod
    def _resize_model_embeddings():
        # This clearly won't work for mulitple tokenizers with unique vocabularies,
        # but it should cover standard use cases (incl. extra vocab on one tokenizer)
        for model in SoftPrompt._models:
            model.resize_token_embeddings(len(SoftPrompt._tokenizers[0]))

    @staticmethod
    def _register_model(model):
        if model not in SoftPrompt._models:
            SoftPrompt._models.append(model)
            # This needs a tokenizer, but if there are none then vocab size hasn't changed.
            # This will be done later when _register_tokenizer() is called.
            if len(SoftPrompt._tokenizers) > 0:
                SoftPrompt._add_tokens_to_tokenizers()
                SoftPrompt._resize_model_embeddings()
                SoftPrompt._refresh_id_lut()

    @staticmethod
    def _register_tokenizer(tokenizer):
        if tokenizer not in SoftPrompt._tokenizers:
            SoftPrompt._tokenizers.append(tokenizer)
            SoftPrompt._add_tokens_to_tokenizers()
            SoftPrompt._resize_model_embeddings()
            SoftPrompt._refresh_id_lut()

    @staticmethod
    def get_special_token_strs():
        """Returns the list of special token strings for all loaded soft prompts.
        """
        special_tokens = list()
        special_tokens.append(SoftPrompt.GENERIC_SOFT_TOKEN_STR)

        for sp in SoftPrompt._soft_prompts:
            special_tokens.append(sp._unique_token_str())

        return special_tokens

    @staticmethod
    def get_special_token_ids():
        """Returns the list of special token ids.
        """
        special_ids = list()
        special_ids.append(SoftPrompt._tokenizers[0].encode(SoftPrompt.GENERIC_SOFT_TOKEN_STR)[0])
        special_ids.extend(SoftPrompt._id_lut.keys())

        return special_ids

    @staticmethod
    def from_file(path: str):
        """Loads a soft prompt from a file.
        """
        with open(path, mode='r') as file:
            j_str = file.read()
            return SoftPrompt.from_json(j_str)

    def to_file(self, path):
        """Save a soft prompt to a path.
        """
        with open(path, mode='w') as file:
            j_str = self.to_json()
            file.write(j_str)

    @staticmethod
    def from_json(string: str):
        """Loads a soft prompt from a serialization.
        """
        j_dict = json.loads(string)

        metadata = j_dict['metadata']
        tensor = pickle.loads(base64.b64decode(j_dict['tensor'].encode('ascii')))
        sp = SoftPrompt(tensor, metadata)
        sp._check_integrity()

        # Check if this soft prompt's uuid already exists
        old_sp = [x for x in SoftPrompt._soft_prompts
                  if x._metadata['uuid'] == x._metadata['uuid']]

        if len(old_sp) != 0:
            sp._metadata['uuid'] = str(uuid.uuid4())

        SoftPrompt._register_soft_prompt(sp)
        return sp

    def to_json(self):
        """Serializes the SoftPrompt to a JSON string representation.
        This can be used to embed the SoftPrompt inside some other file.
        """
        j_dict = dict()
        j_dict['metadata'] = self._metadata
        j_dict['tensor'] = base64.b64encode(pickle.dumps(self._tensor,protocol=4)).decode('ascii')
        return json.dumps(j_dict)

    @staticmethod
    def from_input_id(input_id: int):
        """Gets the already-loaded soft prompt corresponding to a special input_id.

        Returns:
            a SoftPrompt if a corresponding one exists,
            otherwise None
        """
        return SoftPrompt._id_lut.get(input_id)

    @staticmethod
    def from_inputs_embeds(inputs_embeds: torch.Tensor, metadata: Dict[str, Any]=None):
        """Creates a soft prompt from an embedding tensor.
        """
        sp = SoftPrompt(tensor=inputs_embeds.clone(), metadata=metadata)
        sp._check_integrity()
        SoftPrompt._register_soft_prompt(sp)
        return sp

    @staticmethod
    def from_tuning_model(model, metadata: Dict[str, Any]=None):
        """Extracts a soft prompt from a PromptTuningModel.
        """
        return SoftPrompt.from_inputs_embeds(
            inputs_embeds=model.get_soft_params(), metadata=metadata)

    @staticmethod
    def from_string(string: str, model, tokenizer, metadata: Dict[str, Any]=None):
        """Creates a soft prompt by tokenizing and embedding a string.

        This is useful for testing as it is repeatable.
        You can also use this method to set a starting point for training.
        """
        tokens = tokenizer(string, return_tensors="pt").input_ids.to(model.device)

        if metadata is None:
            metadata = dict()
            metadata['name'] = "FromString"
            metadata['description'] = f"Created from string '{string}'"

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

    def get_metadata(self):
        """Returns the metadata.
        """
        return self._metadata