import math
from collections import deque
import os

import numpy as np
import torch
import wandb
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from torch import nn
from tqdm import tqdm

from utils import get_text


class MainStage:
    default_config = {
        'opt_params': {},
        'log_window_size': 10,
        'opt_class': 'Adam',
        'teacher_enforce': {
            'ratio_start': 0.,
            'ratio_growth': 0.,
            'ratio_max': 0.
        }
    }

    def __init__(self, model, stage_name, stage_config, pad_idx):
        self.config = self.default_config.copy()
        self.config.update(stage_config)
        self.name = stage_name
        self.model = model
        self.opt = self.init_opt()
        self.lr_scheduler = self.init_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.teacher_enforce_ratio = self.config['teacher_enforce']['ratio_start']

    def train(self, train_iterator, val_iterator):
        train_step = 0
        val_step = 0
        best_valid_loss = float('inf')
        for epoch in range(self.config['epoch']):
            train_loss, train_step = self.train_epoch(
                train_iterator,
                train_step
            )

            valid_loss, val_step = self.val_epoch(
                val_iterator,
                val_step,
            )

            self.increase_teacher_ratio()
            if valid_loss < best_valid_loss:
                if wandb.run:
                    save_path = os.path.join('model_save', wandb.run.name)
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.name}-model.pt'))
                else:
                    torch.save(self.model.state_dict(), f'{self.name}-model.pt')
                best_valid_loss = valid_loss

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def train_epoch(self, iterator, global_step):
        self.model.train()

        epoch_loss = 0
        loss_window = deque(maxlen=self.config['log_window_size'])
        tqdm_iterator = tqdm(iterator)
        for i, batch in enumerate(tqdm_iterator):
            self.opt.zero_grad()

            loss = self.compute_batch_loss(batch)
            loss.backward()

            # Let's clip the gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.opt.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            epoch_loss += loss.item()
            loss_window.append(loss.cpu().data.numpy())
            if (i + 1) % self.config['log_window_size'] == 0:
                log_dict = dict()
                mean_loss = np.mean(loss_window)
                log_dict['train_loss'] = mean_loss
                log_dict['train_step'] = global_step
                log_dict['learning_rate'] = self.opt.param_groups[0]["lr"]
                log_dict['teacher_ratio'] = self.teacher_enforce_ratio

                if tqdm_iterator._ema_dt():
                    log_dict['train_speed(batch/sec)'] = tqdm_iterator._ema_dn() / tqdm_iterator._ema_dt()
                if wandb.run:
                    wandb.log({self.name: log_dict})
                tqdm_iterator.set_postfix(train_loss=mean_loss)

            global_step += 1

        return epoch_loss / len(iterator), global_step

    def val_epoch(self, iterator, global_step):
        self.model.eval()
        epoch_loss = 0
        loss_window = deque(maxlen=self.config['log_window_size'])
        tqdm_iterator = tqdm(iterator)

        original_text = []
        generated_text = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_iterator):
                loss = self.compute_batch_loss(batch)
                org, gen = self.gen_translate(batch)
                original_text.extend(org)
                generated_text.extend(gen)

                epoch_loss += loss.item()
                loss_window.append(loss.item())

                if (i + 1) % self.config['log_window_size'] == 0:
                    log_dict = dict()
                    mean_loss = np.mean(loss_window)
                    log_dict['val_loss'] = mean_loss
                    log_dict['val_step'] = global_step
                    if wandb.run:
                        wandb.log({self.name: log_dict})
                    tqdm_iterator.set_postfix(val_loss=mean_loss)

                global_step += 1
        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
        if wandb.run:
            wandb.log({self.name: {'bleu': bleu, 'val_step': global_step}})
        print(f'Bleu: {bleu:.3f}')
        return epoch_loss / len(iterator), global_step

    def compute_batch_loss(self, batch, val=False):
        src = batch.src
        trg = batch.trg
        if val:
            output = self.model(src, trg[:-1], 1.)
        else:
            output = self.model(src, trg[:-1], self.teacher_enforce_ratio)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output.view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = self.criterion(output, trg)
        return loss

    def gen_translate(self, batch):
        src = batch.src
        trg = batch.trg
        output = self.model.gen_translate(src, trg[:-1])
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        vocab = self.model.trg_vocab

        org = [get_text(x, vocab) for x in trg.cpu().numpy().T]
        gen = [get_text(x, vocab) for x in output]
        return org, gen

    def increase_teacher_ratio(self):
        growth = self.config['teacher_enforce']['ratio_growth']
        max_ratio = self.config['teacher_enforce']['ratio_max']
        new_ratio = min(self.teacher_enforce_ratio + growth, max_ratio)
        self.teacher_enforce_ratio = new_ratio

    def init_opt(self):
        opt_class = getattr(optim, self.config['opt_class'])
        opt_params = self.config['opt_params']
        opt = opt_class(self.model.parameters(), **opt_params)
        return opt

    def init_scheduler(self):
        # TODO refactor
        if 'scheduler_class' not in self.config:
            return None
        scheduler_class = getattr(optim.lr_scheduler, self.config['scheduler_class'])
        scheduler_params = self.config['scheduler_params']
        scheduler = scheduler_class(self.opt, **scheduler_params)
        return scheduler
