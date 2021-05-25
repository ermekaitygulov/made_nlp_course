import random

import torch
import torch.nn as nn

from neural_network import NN_CATALOG, BaseModel
from utils import add_to_catalog
from beam_search import Node


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=input_dim,
        )

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # embedded = [src sent len, batch size, emb dim]

        rnn_output, (hidden, cell) = self.rnn(embedded)
        # rnn_outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        prediction = self.out(rnn_output)
        output = {
            'rnn_out': rnn_output,
            'rnn_hidden': hidden,
            'rnn_cell': cell,
            'prediction': prediction
        }
        return output


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(Attention, self).__init__()
        self.attn_combine = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.tanh = nn.Tanh()

    def forward(self, enc_outputs, dec_hid):
        # enc_outputs = [seq_len, batch_size, hid_dim]
        # dec_hid = [n layers, batch size, hid_dim]

        enc_outputs = enc_outputs.transpose(0, 1)
        dec_hid = dec_hid.transpose(0, 1)
        # enc_outputs = [batch_size, seq_len, hid_dim]
        # dec_hid = [batch size, n layers, hid_dim]

        attn_scores = torch.bmm(enc_outputs, dec_hid.transpose(1, 2))
        attn_scores = torch.softmax(attn_scores, 1)
        # attn_scores = [batch_size, seq_len, n_layers]

        context = torch.bmm(enc_outputs.transpose(1, 2), attn_scores)
        # context = [batch_size, n_layers, hid_dim]
        hid = torch.cat((context.transpose(1, 2), dec_hid), -1)
        hid = hid.transpose(0, 1).contiguous()
        # hid = [n_layers, batch_size, hid_dim]
        hid = self.attn_combine(hid)
        hid = self.tanh(hid)

        output = {
            'hidden': hid,
            'attention_map': attn_scores,
        }
        return output
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

    def forward(self, x, hidden, cell):
        
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        
        x = x.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(x))
        # embedded = [1, batch size, emb dim]

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(rnn_output.squeeze(0))
        output = {
            'rnn_out': rnn_output,
            'rnn_hidden': hidden,
            'rnn_cell': cell,
            'prediction': prediction
        }
        # prediction = [batch size, output dim]
        
        return output


class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.attention = Attention(hid_dim, hid_dim)
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

    def forward(self, x, hidden, cell, enc_outputs):
        x = x.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(x))
        # embedded = [1, batch size, emb dim]

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        attn_output = self.attention(enc_outputs, hidden)
        hidden = attn_output['hidden']
        prediction = self.out(hidden[-1])
        output = {
            'rnn_out': rnn_output,
            'rnn_hidden': hidden,
            'rnn_cell': cell,
            'prediction': prediction,
            'attention_map': attn_output['attention_map']
        }
        # prediction = [batch size, output dim]

        return output


@add_to_catalog('lstm_enc_dec', NN_CATALOG)
class Seq2Seq(BaseModel):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__(input_dim, output_dim, device, trg_vocab, **kwargs)
        
        self.encoder = Encoder(input_dim, **kwargs['encoder'])
        self.decoder = Decoder(output_dim, **kwargs['decoder'])
        self.teacher_forcing_ratio = kwargs['teacher_forcing_ratio']
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_output = self.encoder(src)
        hidden, cell = enc_output['rnn_hidden'], enc_output['rnn_cell']

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]
        
        for t in range(1, max_len):
            dec_output = self.decoder(dec_input, hidden, cell)
            output, hidden, cell = dec_output['rnn_out'], dec_output['rnn_hidden'], dec_output['rnn_cell']
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.max(1)[1]
            dec_input = (top1 if teacher_force else trg[t])
        
        return outputs


@add_to_catalog('lstm_attn', NN_CATALOG)
class LSTMAttn(BaseModel):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__(input_dim, output_dim, device, trg_vocab, **kwargs)

        self.encoder = Encoder(input_dim, **kwargs['encoder'])
        self.attention = Attention(**kwargs['attention'])
        self.decoder = Decoder(output_dim, **kwargs['decoder'])
        self.teacher_forcing_ratio = kwargs['teacher_forcing_ratio']
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_output = self.encoder(src)
        enc_outputs, hidden, cell = enc_output['rnn_out'], enc_output['rnn_hidden'], enc_output['rnn_cell']

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, max_len):
            dec_output = self.decoder(dec_input, hidden, cell)
            output, hidden, cell = dec_output['prediction'], dec_output['rnn_hidden'], dec_output['rnn_cell']
            hidden = self.attention(enc_outputs, hidden)['hidden']
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            dec_input = (top1 if teacher_force else trg[t])

        return outputs


@add_to_catalog('lstm_dec_attn', NN_CATALOG)
class LSTMDecAttn(BaseModel):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__(input_dim, output_dim, device, trg_vocab, **kwargs)

        self.encoder = Encoder(input_dim, **kwargs['encoder'])
        self.decoder = AttnDecoder(output_dim, **kwargs['decoder'])
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_output = self.encoder(src)
        enc_outputs, hidden, cell = enc_output['rnn_out'], enc_output['rnn_hidden'], enc_output['rnn_cell']

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, max_len):
            dec_output = self.decoder(dec_input, hidden, cell, enc_outputs)
            output, hidden, cell = dec_output['prediction'], dec_output['rnn_hidden'], dec_output['rnn_cell']
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            dec_input = (top1 if teacher_force else trg[t])

        return outputs


@add_to_catalog('lstm_teacher', NN_CATALOG)
class LSTMTeacher(BaseModel):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__(input_dim, output_dim, device, trg_vocab, **kwargs)

        self.encoder = Encoder(input_dim, **kwargs['encoder'])
        self.decoder = AttnDecoder(output_dim, **kwargs['decoder'])
        # self.beam_width = kwargs['beam_width']
        # self.max_length = kwargs['beam_max_length']
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        teacher_start = max_len - int(max_len * teacher_forcing_ratio)
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_output = self.encoder(src)
        enc_outputs, hidden, cell = enc_output['rnn_out'], enc_output['rnn_hidden'], enc_output['rnn_cell']

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]
        for t in range(1, max_len):
            dec_output = self.decoder(dec_input, hidden, cell, enc_outputs)
            output, hidden, cell = dec_output['prediction'], dec_output['rnn_hidden'], dec_output['rnn_cell']
            outputs[t] = output
            teacher_force = t > teacher_start
            top1 = output.max(1)[1]
            dec_input = (top1 if teacher_force else trg[t])

        return outputs

    # def gen_translate(self, src, trg):
    #     batch_size = src.shape[1]
    #     outputs = []
    #     for i in range(batch_size):
    #         outputs.append(self.beam_search(src[:, i:i+1], self.beam_width, 1, max_length=self.max_length)[0])
    #     return outputs
    #
    # # beam search
    # def beam_search(self, src, beam_width=4, num_hypotheses=1, max_length=500):
    #     # last hidden state of the encoder is used as the initial hidden state of the decoder
    #     enc_output = self.encoder(src)
    #     memory, hidden, cell = enc_output['rnn_out'], enc_output['rnn_hidden'], enc_output['rnn_cell']
    #
    #     fringe = [Node(parent=None, state=(hidden, cell),
    #                    value=torch.tensor(self.sos_idx).to(self.device), cost=0.0)]
    #     hypotheses = []
    #
    #     for _ in range(max_length):
    #         cur_token = torch.cat([n.value.unsqueeze(0) for n in fringe])
    #         hidden_batch = torch.cat([n.state[0] for n in fringe], dim=1)
    #         cell_batch = torch.cat([n.state[1] for n in fringe], dim=1)
    #         memory_batch = torch.cat([memory for _ in fringe], dim=1)
    #
    #         dec_output = self.decoder(cur_token, hidden_batch, cell_batch, memory_batch)
    #         log_p_batch = torch.log_softmax(dec_output['prediction'], -1)
    #         hidden_batch = dec_output['rnn_hidden']
    #         cell_batch = dec_output['rnn_cell']
    #
    #         next_token = torch.argsort(log_p_batch, dim=-1, descending=True)[:, :beam_width]
    #
    #         for i, (token_list, log_p, parent) in enumerate(zip(next_token, log_p_batch, fringe)):
    #             hid = hidden_batch[:, i:i+1]
    #             cell = cell_batch[:, i:i+1]
    #
    #             for candidate, candidate_score in zip(token_list, log_p):
    #                 n_new = Node(parent=parent, state=(hid, cell), value=candidate, cost=candidate_score)
    #                 if n_new.value == self.eos_idx:
    #                     hypotheses.append(n_new)
    #                 else:
    #                     fringe.append(n_new)
    #         if not fringe:
    #             break
    #         fringe = sorted(fringe, key=lambda node: node.cum_cost / node.length, reverse=True)[:beam_width]
    #     hypotheses.extend(fringe)
    #     hypotheses.sort(key=lambda node: node.cum_cost / node.length)
    #     return [node.to_sequence_of_values() for node in hypotheses[:num_hypotheses]]
