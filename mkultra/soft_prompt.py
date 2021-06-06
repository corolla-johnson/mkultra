import json
import uuid

class SoftPrompt:
    """
    A soft prompt.

    After loading any number of soft prompts, make sure to prime your tokenizer with
    tokenizer.add_special_tokens(SoftPrompt.get_special_tokens()).

    Attributes:
        _tensor: The underlying tensor.
        _metadata: A metadata dictionary. Keys are:
                   'name', 'uuid', 'description', 'date', 'length'
    """
    # Dictionary mapping special tokens to loaded soft prompts
    _soft_prompts = dict()

    def __init__(self, tensor, metadata=None):
        self._tensor = tensor
        self._metadata = metadata

    def __len__(self):
        return self._metadata['length']

    def _check_metadata(self):
        if self._metadata.get('name') is None:
            self._metadata['name'] = "untitled"

        if self._metadata.get('uuid') is None:
            self.

    @staticmethod
    def get_special_tokens():
        """Returns the list of special tokens for all loaded soft prompts.
        """

    @staticmethod
    def from_file(path):
        """Loads a soft prompt from a file.
        """
        with open(path) as file:
            sp = json.load(file)
            sp._check_metadata()

            # Check if this soft prompt is already loaded
            if sp._metadata['uuid'] ==

    @staticmethod
    def from_special_token(id):
        """Gets the already-loaded soft prompt corresponding to a special input_id.

        Returns:
            a SoftPrompt if a corresponding one exists,
            otherwise None
        """
        return SoftPrompt._loaded_soft_prompts.get(id)

    @staticmethod
    def from_inputs_embeds(inputs_embeds, metadata=None):
        """Creates a soft prompt from an embedding tensor.
        """
        pass

    @staticmethod
    def from_tuning_model(model, metadata=None):
        """Extracts a soft prompt from a PromptTuningModel.
        """
        pass

    @staticmethod
    def from_string(string, model, tokenizer, metadata=None):
        """Creates a soft prompt by tokenizing and embedding a string.

        This is useful for testing as it is repeatable.
        You can also use this method to set a starting point for training.
        """
        pass

    def get_tag_str():
        """Returns a string used to mark a location for this soft prompt.

        This will consist of unique special token containing the prompt's name and GUID,
        followed by a number of '<@>'s that represent individual soft tokens.
        """
        pass

    def get_input_ids():
        """Returns a tensor of input ids used to mark a location for this soft prompt.

        This will consist of unique special token containing the prompt's name and GUID,
        followed by a number of '<@>'s that represent individual soft tokens.
        """
        pass

    def get_special_token():

    def get_inputs_embeds():
        """Returns the embedding tensor of this soft prompt.
        """
        pass

