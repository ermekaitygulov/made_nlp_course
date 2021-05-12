from collections import deque

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch
from torch import nn
import wandb
from homeworks.Lab02_NMT.utils import get_text


def train(model, iterator, optimizer, criterion, clip, global_step=0, log_window_size=10):
    model.train()

    epoch_loss = 0
    loss_window = deque(maxlen=log_window_size)
    tqdm_iterator = tqdm(iterator)
    for i, batch in enumerate(tqdm_iterator):
        optimizer.zero_grad()

        src = batch.src
        trg = batch.trg

        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()

        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        loss_window.append(loss.cpu().data.numpy())
        if (i + 1) % log_window_size == 0:
            log_dict = dict()
            mean_loss = np.mean(loss_window)
            log_dict['train_loss'] = mean_loss
            log_dict['train_step'] = global_step
            if tqdm_iterator._ema_dt():
                log_dict['train_speed(batch/sec)'] = tqdm_iterator._ema_dn() / tqdm_iterator._ema_dt()
            wandb.log(log_dict)
            tqdm_iterator.set_postfix(train_loss=mean_loss)

        global_step += 1

    return epoch_loss / len(iterator), global_step


def evaluate(model, iterator, criterion, global_step=0, log_window_size=10):
    model.eval()
    epoch_loss = 0
    loss_window = deque(maxlen=log_window_size)
    tqdm_iterator = tqdm(iterator)
    with torch.no_grad():

        for i, batch in enumerate(tqdm_iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            loss_window.append(loss.item())

            if (i + 1) % log_window_size == 0:
                log_dict = dict()
                mean_loss = np.mean(loss_window)
                log_dict['val_loss'] = mean_loss
                log_dict['val_step'] = global_step
                wandb.log(log_dict)
                tqdm_iterator.set_postfix(train_loss=mean_loss)

            global_step += 1

    return epoch_loss / len(iterator), global_step


def test(model, test_iterator, vocab, total_step):
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, vocab) for x in output[1:].detach().cpu().numpy().T])
    bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
    table = wandb.Table(data=[[total_step, bleu]], columns=["total_step", "bleu"])
    wandb.log({"my_custom_plot_id": wandb.plot.scatter(
        table,
        "total_step",
        "bleu", title="Custom Y vs X Scatter Plot")
    })


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    # <YOUR CODE HERE>
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
