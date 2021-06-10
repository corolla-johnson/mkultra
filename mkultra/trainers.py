from transformers import Trainer
from transformers import Adafactor

class SoftPromptTrainer(Trainer):
    """ Extends huggingface trainer for use with mkultra tuning models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        # Don't support larger batches
        self.args.per_device_train_batch_size = 1
        return super().train(*args, **kwargs)


class WorldInfoTrainer:
    pass