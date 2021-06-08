from transformers import Trainer
from transformers import Adafactor

class SoftPromptTrainer(Trainer):
    """
    """
    pass

    def create_optimizer(self):
        self.optimizer = Adafactor(self.model.get_prompt_tuning_params())

class WorldInfoTrainer:
    pass