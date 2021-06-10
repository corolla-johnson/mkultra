from transformers import Trainer
from transformers import Adafactor
import random
import torch

class WorldInfoTrainer:
    def __init__(self, model, tokenizer, optimizer, blocks, max_spacing=0, max_block_size=1024, min_loss=0):
        self.model = model
        self.tokenizer = tokenizer
        if optimizer is None:
            self.optimizer = Adafactor([model.get_soft_params()])
        else:
            self.optimizer = optimizer
        self.blocks = blocks
        self.max_spacing = max_spacing
        self.max_block_size = max_block_size
        self.min_loss = min_loss

        self.tokenize_blocks()

    def tokenize_blocks(self):
        for block in self.blocks:
            block['call'] = self.tokenizer(block['call'], return_tensors="pt").input_ids.to(self.model.device)
            block['response'] = self.tokenizer(block['response'], return_tensors="pt").input_ids.to(self.model.device)

    def arrange_blocks(self):
        arranged_blocks = list()

        for block in self.blocks:
            call = block['call']
            response = block['response']
            real_max_spacing = min(
                self.max_block_size - self.model.learned_embedding.shape[-2] - call.shape[-1] - response.shape[-1],
                self.max_spacing)

            #breakpoint()

            spacing = random.randint(0, real_max_spacing)
            space_ids = torch.randint(low=0, high=len(self.tokenizer), size=(1, spacing)).to(self.model.device)

            #breakpoint()

            ignore_len = call.shape[-1] + spacing

            #breakpoint()


            input_ids = torch.cat([space_ids, call, response], dim=1)
            labels = torch.cat([torch.full((1,ignore_len),-100).to(self.model.device), response], dim=1)

            #breakpoint()

            arranged_blocks.append((input_ids, labels))

        random.shuffle(arranged_blocks)

        return arranged_blocks

    def train(self, epochs=1):
        self.model.train()

        steps = 0
        total_steps = len(self.blocks) * epochs

        for _ in range(epochs):
            arranged_blocks = self.arrange_blocks()

            for input_ids, labels in arranged_blocks:
                self.optimizer.zero_grad()
                output = self.model(input_ids=input_ids, labels=labels)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                print(f"{steps}/{total_steps}: Loss: {loss}")
                steps += 1

                if(loss < self.min_loss):
                    return