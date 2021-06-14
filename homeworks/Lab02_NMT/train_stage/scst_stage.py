import os

import torch
import wandb
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

from train_stage.main_stage import MainStage
from utils import get_text


class SCSTStage(MainStage):
    def __init__(self, model, stage_name, stage_config, pad_idx,
                 train_translator, src_vocab, trg_vocab):
        super(SCSTStage, self).__init__(model, stage_name, stage_config, pad_idx)
        self.train_translator = train_translator
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def train(self, train_iterator, val_iterator):
        train_step = 0
        val_step = 0
        best_val_bleu = 0.
        for epoch in range(self.config['epoch']):
            train_loss, train_step = self.train_epoch(
                train_iterator,
                train_step
            )

            val_bleu, val_loss, val_step = self.val_epoch(
                val_iterator,
                val_step,
            )

            self.increase_teacher_ratio()
            if val_bleu > best_val_bleu:
                if wandb.run:
                    save_path = os.path.join('model_save', wandb.run.name)
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.name}-model.pt'))
                else:
                    torch.save(self.model.state_dict(), f'{self.name}-model.pt')
                best_val_bleu = val_bleu

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tVal BLEU: {val_bleu:.3f}')

    def val_epoch(self, iterator, global_step):
        self.model.eval()
        tqdm_iterator = tqdm(iterator)

        original_text = []
        generated_text = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_iterator):
                org, gen = self.gen_translate(batch)
                original_text.extend(org)
                generated_text.extend(gen)
                global_step += 1

        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
        if wandb.run:
            wandb.log({self.name: {'bleu': bleu, 'val_step': global_step}})
        return bleu, None, global_step

    def compute_batch_loss(self, batch, val=False):
        """ Compute pseudo-loss for policy gradient given a batch of sources """
        src = batch.src
        trg = batch.trg

        # use model to __sample__ symbolic translations given input_sequence
        sample_logp, sample_translations = self.model.gen_translate(src, trg[:-1], greedy=False)  # YOUR CODE
        # use model to __greedy__ symbolic translations given input_sequence
        with torch.no_grad():
            greedy_logp, greedy_translations = self.model.gen_translate(src, trg[:-1], greedy=True)  # YOUR CODE

        # compute rewards and advantage
        # be careful with the device, rewards require casting to numpy, so send everything to cpu
        rewards = self.compute_reward(src.cpu(), sample_translations.cpu())
        baseline = self.compute_reward(src.cpu(), greedy_translations.cpu())
        # compute advantage using rewards and baseline
        # be careful with the device, advantage is used to compute gradients, so send it to device
        advantage = (rewards - baseline).to(self.model.device)  # YOUR CODE

        # compute log_pi(a_t|s_t), shape = [batch, seq_length]
        logp_sample = torch.sum(
            to_one_hot(
                sample_translations,
                n_dims=self.model.output_dim
            ) * sample_logp,
            dim=-1
        )
        # YOUR CODE

        # ^-- hint: look at how crossentropy is implemented in supervised learning loss above
        # mind the sign - this one should not be multiplied by -1 :)

        # policy gradient pseudo-loss. Gradient of J is exactly policy gradient.
        J = logp_sample * advantage[None, :]

        # average with mask
        mask = infer_mask(sample_translations, self.model.eos_idx, batch_first=False)
        loss = - torch.sum(J * mask) / torch.sum(mask)

        # regularize with negative entropy. Don't forget the sign!
        # note: for entropy you need probabilities for all tokens (sample_logp), not just logp_sample
        entropy = -torch.sum(torch.exp(sample_logp) * sample_logp, dim=-1)
        entropy = torch.sum(entropy * mask) / torch.sum(mask)
        # <compute entropy matrix of shape[batch, seq_length], H = -sum(p*log_p), don't forget the sign!>
        # hint: you can get sample probabilities from sample_logp using math :)
        reg = - self.config['entropy_weight'] * entropy
        return {
            'loss': loss + reg,
            'entropy': entropy,
            'baseline': torch.mean(baseline) * 100,
            'rewards': torch.mean(rewards) * 100,
        }

    def compute_reward(self, source_idx, translations):
        source_text = [get_text(x, self.src_vocab) for x in source_idx.numpy().T]
        target_text = [self.train_translator[' '.join(s)] for s in source_text]
        translation_text = [get_text(x, self.trg_vocab) for x in translations.numpy().T]
        distances = [sentence_bleu(ref_list, hyp) for ref_list, hyp in zip(target_text, translation_text)]
        return torch.tensor(distances, dtype=torch.float32)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.to(dtype=torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def infer_mask(seq, eos_ix, batch_first=True, include_eos=True, dtype=torch.float):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if batch_first else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: lengths, int32 vector of shape [batch]
    """
    assert seq.dim() == 2
    is_eos = torch.eq(seq, eos_ix).to(dtype=torch.float)
    if include_eos:
        if batch_first:
            is_eos = torch.cat((is_eos[:,:1]*0, is_eos[:, :-1]), dim=1)
        else:
            is_eos = torch.cat((is_eos[:1,:]*0, is_eos[:-1, :]), dim=0)
    count_eos = torch.cumsum(is_eos, dim=1 if batch_first else 0)
    mask = torch.eq(count_eos, 0)
    return mask.to(dtype=dtype)
