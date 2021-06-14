from collections import deque

import torch
import wandb
from tqdm import tqdm
import numpy as np

from train_stage.main_stage import MainStage


class PretrainStage(MainStage):
    def val_epoch(self, iterator, global_step):
        self.model.eval()
        epoch_loss = 0
        loss_window = deque(maxlen=self.config['log_window_size'])
        tqdm_iterator = tqdm(iterator)

        with torch.no_grad():
            for i, batch in enumerate(tqdm_iterator):
                loss = self.compute_batch_loss(batch)['loss']

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
        return 0., epoch_loss / len(iterator), global_step

    def compute_batch_loss(self, batch, val=False):
        src = batch.src

        output = self.model.encoder(src[:-1])['prediction']  # turn off teacher forcing

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = src[2:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = self.criterion(output, trg)
        return {'loss': loss}
