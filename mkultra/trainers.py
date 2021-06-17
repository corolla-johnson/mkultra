from transformers import Trainer
from transformers import Adafactor
import random
import torch

class WorldInfoTrainer:
    def __init__(self, model, tokenizer, optimizer, blocks, max_spacing=0, max_block_size=1024, min_loss=0, repeat_to_fill=True):
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
        self.repeat_to_fill = repeat_to_fill

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

            spacing = random.randint(0, real_max_spacing)
            space_ids = torch.randint(low=0, high=len(self.tokenizer), size=(1, spacing)).to(self.model.device)

            ignore_len = call.shape[-1] + spacing

            # Cat spacing and call first
            input_ids = torch.cat([space_ids, call], dim=1)
            labels = torch.cat([torch.full((1,ignore_len),-100).to(self.model.device)], dim=1)

            if self.repeat_to_fill:
                # Cat response until nearly full
                while (input_ids.shape[-1] + response.shape[-1]) < self.max_block_size:
                    input_ids = torch.cat([input_ids, response], dim=1)
                    labels = torch.cat([labels, response], dim=1)
                    print(input_ids.shape)
            else:
                input_ids = torch.cat([input_ids, response], dim=1)
                labels = torch.cat([labels, response], dim=1)

            arranged_blocks.append((input_ids, labels))

        random.shuffle(arranged_blocks)

        return arranged_blocks

    def train(self, epochs=1):
        self.model.train()

        steps = 0
        total_steps = len(self.blocks) * epochs

        for i in range(epochs):
            arranged_blocks = self.arrange_blocks()

            epoch_loss = 0

            for input_ids, labels in arranged_blocks:
                self.optimizer.zero_grad()
                output = self.model(input_ids=input_ids, labels=labels)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                steps += 1
                epoch_loss += loss.item()

            epoch_loss /= len(arranged_blocks)
            print(f"Epoch {i} loss: {epoch_loss}")

            if(epoch_loss < self.min_loss):
                return

class SoftPromptTrainer:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 project_dir=None,
                 text_path=None,
                 block_size=32,
                 n_tokens=20,
                 ema_alpha=0.1,
                 checkpoint_interval=200,
                 shuffle_seed=None):

        self.model=model
        self.optimizer=optimizer
        self.project_dir=project_dir
        self.text_path=text_path
        self.block_size=block_size
        self.n_tokens=n_tokens
        self.ema_alpha=ema_alpha
        self.checkpoint_interval=checkpoint_interval
        self.shuffle_seed=shuffle_seed

        self._maybe_create_project_directory()
        self.loaded_sp = self._load_latest_checkpoint()
        self.tokens = self._get_tokens()
        self.blocks = self._get_blocks()
        self.block_order = self._get_block_order()

        # Initialize soft prompt in model
        if self.loaded_sp is None:
            model.initialize_soft_prompt(n_tokens=n_tokens)
            self.sp_step = 0
            self.ema_loss = None
            self.eval_loss = None
        else:
            model.set_soft_prompt(self.loaded_sp)
            self.sp_step = self.loaded_sp._metadata['step']
            self.ema_loss = self.loaded_sp._metadata['loss']
            self.eval_loss = self.loaded_sp._metadata['eval_loss']

    def _filename_for_checkpoint(self, step):
        return f"{self._project_name()}-step-{step}.json"

    def _project_name(self):
        import os
        return os.path.basename(os.path.normpath(self.project_dir))

    def _maybe_create_project_directory(self):
        from mkultra.soft_prompt import SoftPrompt
        import os
        # Look for existing project directory
        try:
            os.makedirs(self.project_dir)
            print(f"Created project directory at {self.project_dir}")
        except FileExistsError:
            print(f"Found project directory at {self.project_dir}")

    def _load_latest_checkpoint(self):
        from mkultra.soft_prompt import SoftPrompt
        import os

        # Look for existing checkpoints
        project_files = os.listdir(self.project_dir)
        if project_files is not None:
            checkpoint_files = [check_file for check_file in project_files if ('-step-' in check_file) ]
            if len(checkpoint_files) > 0:
                highest_step = max([ int(check_file[check_file.rfind('-step-')+6:-5]) for check_file in checkpoint_files ])
                print(f"Loading latest checkpoint: {highest_step}")
                return SoftPrompt.from_file( os.path.join(self.project_dir, self._filename_for_checkpoint(highest_step)) )
            else:
                print("No checkpoints found")

        return None

    def _get_tokens(self):
        import json
        import os
        from transformers import GPT2TokenizerFast

        tokens = None
        tokens_path = os.path.join(self.project_dir,"tokens.json")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # See if we already have a tokens file
        try:
            with open(tokens_path, 'r', encoding='utf-8') as file:
                tokens = json.load(file)
                print("Loaded existing tokens.json file")
        except FileNotFoundError:
            print("No tokens.json exists, creating it...")

        # If not, make one from the text path
        if tokens is None:
            with open(self.text_path, 'r', encoding='utf-8') as file:
                text = file.read()
            tokens = tokenizer.encode(text)
            with open(tokens_path, 'x', encoding='utf-8') as file:
                json.dump(tokens, file)

        return tokens

    def _get_blocks(self):
        import math

        # Partition tokens into blocks
        blocks = list()
        num_blocks = math.ceil(len(self.tokens)/self.block_size)

        for block_num in range(num_blocks):
            start = block_num * self.block_size
            end = min(start + self.block_size, len(self.tokens))
            blocks.append( self.tokens[start:end] )

        return blocks

    def _get_block_order(self):
        import os
        import json

        block_order_path = os.path.join(self.project_dir, "block_order.json")

        # See if we already have a block_order file
        try:
            with open(block_order_path, 'r', encoding='utf-8') as file:
                block_order = json.load(file)
                print("Loaded existing block_order.json file")

        except FileNotFoundError:
            print("No block_order.json exists, creating it...")
            block_order = [*range(len(self.blocks))]

            with open(block_order_path, 'x', encoding='utf-8') as file:
                json.dump(block_order, file)

        return block_order

    def train(self, num_training_steps=None):
        from tqdm import tqdm
        from mkultra.soft_prompt import SoftPrompt
        import random
        import torch
        import os
        import json
        import math

        # Train for one epoch by default
        if num_training_steps is None:
            num_training_steps = len(self.blocks)

        self.model.train()
        torch.cuda.empty_cache()
        loss_log_path = os.path.join(self.project_dir,"loss_log.csv")
        bar = tqdm(total=num_training_steps)
        session_step = 0

        # If we have a shuffle seed, shuffle beforehand
        if self.shuffle_seed is not None:
            random.seed(self.shuffle_seed)
            self.block_order = [*range(len(self.blocks))]
            random.shuffle(self.block_order)

        while session_step < num_training_steps:
            # Shuffle blocks every epoch
            if self.sp_step % len(self.blocks) == 0:
                random.shuffle(self.block_order)

                with open(os.path.join(self.project_dir, "block_order.json"), 'w', encoding='utf-8') as file:
                    json.dump(self.block_order, file)

            idx = self.sp_step % len(self.blocks)
            block = self.blocks[self.block_order[idx]]

            input_ids = torch.LongTensor(block).unsqueeze(0).cuda().detach()

            # Forward pass and optimize
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            instant_loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Discard tensor that was moved to GPU
            del input_ids
            torch.cuda.empty_cache()

            if math.isnan(instant_loss):
                raise ValueError(f"NaN loss at step {self.sp_step}")

            # Calculate EMA loss
            self.ema_loss = self.ema_alpha*instant_loss + (1-self.ema_alpha)*self.ema_loss if self.ema_loss is not None else instant_loss

            bar.set_postfix({
                "Model Step" : self.sp_step,
                "EMA Loss"   : self.ema_loss,
            })
            bar.update(1)

            # Save checkpoint every so often
            if self.sp_step%self.checkpoint_interval == 0:
                sp = SoftPrompt.from_tuning_model(self.model,
                    {"name"     : f"{self._project_name} Step {self.sp_step}",
                    "step"      : self.sp_step,
                    "loss"      : self.ema_loss})
                sp.to_file( os.path.join( self.project_dir,self._filename_for_checkpoint(self.sp_step) ) )

            with open(loss_log_path, 'a', encoding='utf-8') as file:
                file.write(f"{self.sp_step},{self.ema_loss}\n")

            session_step += 1
            self.sp_step += 1

        # Save a checkpoint once done
        sp = SoftPrompt.from_tuning_model(self.model,
            {"name"  : f"{self._project_name} {self.sp_step}",
            "step"  : self.sp_step,
            "loss"  : self.ema_loss})
        sp.to_file( os.path.join( self.project_dir,self._filename_for_checkpoint(self.sp_step) ) )

    def evaluate(self, eval_percentage=0.1):
        from tqdm import tqdm
        import torch

        self.model.eval()
        eval_steps = round(eval_percentage * len(self.blocks))
        bar = tqdm(total=eval_steps)
        session_step = 0

        # If we have a shuffle seed, shuffle beforehand
        if self.shuffle_seed is not None:
            random.seed(self.shuffle_seed-1)
            self.block_order = [*range(len(self.blocks))]
            random.shuffle(self.block_order)

        eval_loss = 0

        while session_step < eval_steps:
            block = self.blocks[self.block_order[session_step]]

            input_ids = torch.LongTensor(block).unsqueeze(0).cuda().detach()

            with torch.no_grad():
                # Forward pass and optimize
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss.item()

            eval_loss += loss

            # Discard tensor that was moved to GPU
            del input_ids
            torch.cuda.empty_cache()

            bar.set_postfix({
                "Loss"   : loss,
            })
            bar.update(1)
            session_step += 1

        eval_loss /= eval_steps

        return eval_loss