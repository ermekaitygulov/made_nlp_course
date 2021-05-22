import math
from collections import deque

import numpy as np
import torch
import wandb
from torch import optim
from torch import nn
from tqdm import tqdm


class MainStage:
    default_config = {
        'opt_params': {},
        'log_window_size': 10,
        'opt_class': 'Adam'
    }

    def __init__(self, model, stage_name, stage_config, pad_idx):
        self.config = self.default_config.copy()
        self.config.update(stage_config)
        self.name = stage_name
        self.model = model
        self.opt = self.init_opt()
        self.lr_scheduler = self.init_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

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

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{self.name}-model.pt')

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

        with torch.no_grad():
            for i, batch in enumerate(tqdm_iterator):
                loss = self.compute_batch_loss(batch)
                epoch_loss += loss.item()
                loss_window.append(loss.item())

                if (i + 1) % self.config['log_window_size'] == 0:
                    log_dict = dict()
                    mean_loss = np.mean(loss_window)
                    log_dict['val_loss'] = mean_loss
                    log_dict['val_step'] = global_step
                    if wandb.run:
                        wandb.log({self.name: log_dict})
                    tqdm_iterator.set_postfix(train_loss=mean_loss)

                global_step += 1

        return epoch_loss / len(iterator), global_step

    def compute_batch_loss(self, batch):
        src = batch.src
        trg = batch.trg

        output = self.model(src, trg)  # turn off teacher forcing

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = self.criterion(output, trg)
        return loss

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
