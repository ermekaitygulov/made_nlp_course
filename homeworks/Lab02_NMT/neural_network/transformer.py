import math
import random

import torch
import torch.nn as nn

from neural_network import NN_CATALOG, BaseModel
from neural_network.lstm import AttnDecoder
from utils import add_to_catalog


@add_to_catalog('transformer', NN_CATALOG)
class Transformer(BaseModel):
    def __init__(self, input_dim, output_dim, device, pad_idx, emb_dim, **kwargs):
        super(Transformer, self).__init__(input_dim, output_dim, device, pad_idx, **kwargs)
        self.enc_embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        self.dec_embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        self.pos_encoding = PositionalEncoding(emb_dim)

        enc_layer_n = kwargs['encoder'].pop('layer_number')
        encoder_layer = nn.TransformerEncoderLayer(emb_dim, **kwargs['encoder'])
        encoder_norm = nn.LayerNorm(emb_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, enc_layer_n, encoder_norm)

        dec_layer_n = kwargs['decoder'].pop('layer_number')
        decoder_layer = nn.TransformerDecoderLayer(emb_dim, **kwargs['decoder'])
        decoder_norm = nn.LayerNorm(emb_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, dec_layer_n, decoder_norm)

        self.out = nn.Linear(emb_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.):
        src_pad_mask = src == self.pad_idx
        trg_pad_mask = trg == self.pad_idx
        trns_kwargs = {
            'src_key_padding_mask': src_pad_mask.transpose(0, 1),
            'memory_key_padding_mask': src_pad_mask.transpose(0, 1),
            'trg_key_padding_mask': trg_pad_mask.transpose(0, 1)
        }
        batch_size = trg.shape[1]
        trg_vocab_size = self.output_dim
        max_len = trg.shape[0]
        teacher_start = max_len - int(max_len * teacher_forcing_ratio)

        src_emb = self.enc_embedding(src)
        src_emb = self.pos_encoding(src_emb)

        memory = self.encoder(
            src_emb,
            src_key_padding_mask=trns_kwargs['src_key_padding_mask'],
        )

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        if self.training:
            trg_emb = self.dec_embedding(trg[:teacher_start - 1])
            trg_emb = self.pos_encoding(trg_emb)
            tgt_mask = self.generate_square_subsequent_mask(teacher_start - 1).to(self.device)
            dec_out = self.decoder(trg_emb, memory,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=trns_kwargs['trg_key_padding_mask'][:, :teacher_start - 1],
                                   memory_key_padding_mask=trns_kwargs['memory_key_padding_mask']
                                   )
            dec_outputs = self.out(dec_out)
            outputs[1:teacher_start] = dec_outputs

            if teacher_forcing_ratio:
                gen_outputs = self.generate_seq(trg, memory, trns_kwargs, teacher_start)
                outputs[teacher_start:] = gen_outputs
        else:
            outputs[1:] = self.generate_seq(trg, memory, trns_kwargs)
        return outputs

    def generate_seq(self, trg, memory, trns_kwargs, start_idx=1):
        max_len = trg.shape[0]
        outputs = []

        dec_input = trg[:start_idx]
        tgt_mask = self.generate_square_subsequent_mask(max_len).to(self.device)

        for i in range(start_idx, max_len):
            trg_emb = self.dec_embedding(dec_input)
            trg_emb = self.pos_encoding(trg_emb)
            dec_out = self.decoder(trg_emb, memory,
                                   tgt_mask=tgt_mask[:i, :i],
                                   tgt_key_padding_mask=trns_kwargs['trg_key_padding_mask'][:, :i],
                                   memory_key_padding_mask=trns_kwargs['memory_key_padding_mask']
                                   )
            out = self.out(dec_out)
            outputs.append(out[-1])
            top1 = out[-1].max(-1)[1]
            top1 = top1.unsqueeze(0)
            dec_input = torch.cat([dec_input, top1])
        outputs = torch.stack(outputs)
        return outputs

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


@add_to_catalog('hybrid', NN_CATALOG)
class Hybrid(BaseModel):
    def __init__(self, input_dim, output_dim, device, pad_idx, emb_dim, **kwargs):
        super(Hybrid, self).__init__(input_dim, output_dim, device, pad_idx, **kwargs)
        self.enc_embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        self.pos_encoding = PositionalEncoding(emb_dim)

        enc_layer_n = kwargs['encoder'].pop('layer_number')
        encoder_layer = nn.TransformerEncoderLayer(emb_dim, **kwargs['encoder'])
        encoder_norm = nn.LayerNorm(emb_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, enc_layer_n, encoder_norm)

        self.decoder = AttnDecoder(output_dim, **kwargs['decoder'])
        self.out = nn.Linear(emb_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.):
        src_pad_mask = src == self.pad_idx
        trns_kwargs = {
            'src_key_padding_mask': src_pad_mask.transpose(0, 1),
        }
        batch_size = trg.shape[1]
        trg_vocab_size = self.output_dim
        max_len = trg.shape[0]
        teacher_start = max_len - int(max_len * teacher_forcing_ratio)

        src_emb = self.enc_embedding(src)
        src_emb = self.pos_encoding(src_emb)

        memory = self.encoder(
            src_emb,
            src_key_padding_mask=trns_kwargs['src_key_padding_mask'],
        )

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden = memory[-1].unsqueeze(0)
        cell = torch.zeros_like(hidden)

        dec_input = trg[0, :]
        for t in range(1, max_len):
            dec_output = self.decoder(dec_input, hidden, cell, memory)
            output, hidden, cell = dec_output['prediction'], dec_output['rnn_hidden'], dec_output['rnn_cell']
            outputs[t] = output
            teacher_force = t > teacher_start
            top1 = output.max(1)[1]
            dec_input = (top1 if teacher_force else trg[t])

        return outputs


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)