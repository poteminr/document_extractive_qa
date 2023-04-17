import os
import random
import torch
import numpy as np
import wandb
import evaluate
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from transformers import set_seed, default_data_collator, get_scheduler
from utils.checkpoint_saver import CheckpointSaver
from utils.dataset import DocumentDataset


class TrainerConfig:
    epochs = 20
    lr = 1e-4
    batch_size = 32
    betas = (0.9, 0.95)
    clip_gradients = False
    grad_norm_clip = 10
    num_workers = 0
    lr_scheduler = False
    num_training_steps_for_freeze = 3000 # -1 to train whole model from first steps

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, config: TrainerConfig, train_dataset: DocumentDataset,
                 val_dataset: Optional[DocumentDataset] = None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric = evaluate.load("squad")
        self.seed = 1007
        self.seed_everything(self.seed)
        self.checkpoint_saver = CheckpointSaver(dirpath='./model_weights', model_type=self.config.model_type)

    def create_dataloader(self, dataset: DocumentDataset, shuffle=False):
        return DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=shuffle,
                          collate_fn=default_data_collator, num_workers=self.config.num_workers)

    def train(self):
        self.seed_everything(self.seed)
        model = self.model.to(self.device)
        wandb.watch(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, betas=self.config.betas)

        train_loader = self.create_dataloader(self.train_dataset, shuffle=True)
        if self.val_dataset is not None:
            val_loader = self.create_dataloader(self.val_dataset)

        if self.config.lr_scheduler:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=self.config.epochs * len(train_loader),
            )
            
        is_encoder_frozen = False
        if self.config.num_training_steps_for_freeze != -1:
            self.set_encoder_requires_grad(model, requires_grad=False)
            is_encoder_frozen = True

        num_training_steps = 0
        for epoch in range(self.config.epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            average_loss = 0
            for step, batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)

                if num_training_steps >= self.config.num_training_steps_for_freeze and is_encoder_frozen:
                    is_encoder_frozen = False
                    self.set_encoder_requires_grad(model, requires_grad=True)
                
                outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions
                            )
        
                loss = outputs.loss
                average_loss += loss.item()
                num_training_steps += batch['input_ids'].size()[0]
                pbar.set_description(f"epoch {epoch + 1} iter {step} | train_loss: {loss.item():.5f}")

                loss.backward()
                if self.config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_gradients)

                optimizer.step()
                if self.config.lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()

            average_loss = average_loss / len(train_loader)
            wandb.log({f'train_loss': average_loss}, step=epoch + 1)
            print(f"train_loss: {average_loss}, epoch: {epoch + 1}")

            if self.val_dataset is not None:
                # Evaluation
                model.eval()
                start_logits = []
                end_logits = []
                average_val_loss = 0
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    start_positions = batch['start_positions'].to(self.device)
                    end_positions = batch['end_positions'].to(self.device)
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions
                            )

                    average_val_loss += outputs['loss'].item()
                    start_logits.append(outputs.start_logits.cpu().numpy())
                    end_logits.append(outputs.end_logits.cpu().numpy())

                start_logits = np.concatenate(start_logits)[: len(self.val_dataset)]
                end_logits = np.concatenate(end_logits)[: len(self.val_dataset)]

                average_val_loss = average_val_loss / len(val_loader)
                metrics = self.compute_metrics(start_logits, end_logits, self.val_dataset)
                metrics.update({'val_loss': average_val_loss})
                wandb.log(metrics, step=epoch + 1)
                print(f"epoch {epoch + 1}:", metrics)
                self.checkpoint_saver(model, epoch + 1, metrics['exact_match'])

        wandb.finish()

    def compute_metrics(self, start_logits, end_logits, dataset, n_best: int = 20, max_answer_length: int = 310):
        predicted_answers = []
        for idx, example in enumerate(dataset):
            example_id = example['id']
            context = dataset.get_context(example_id.item())
            answers = []

            start_logit = start_logits[idx]
            end_logit = end_logits[idx]
            offsets = example['offset_mapping']

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answers.append(
                        {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": str(example_id.item()), "prediction_text": best_answer["text"]})

        theoretical_answers = [{"id": str(ex["id"].item()), "answers": dataset.get_answer(ex["id"].item())} for ex in
                               dataset]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)

    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        set_seed(seed)
    
    @staticmethod
    def set_encoder_requires_grad(model, requires_grad=False):
        encoder, _ = model.children()
        for param in encoder.parameters():
            param.requires_grad = requires_grad
        print(f'Encoder grad set to {requires_grad}')
