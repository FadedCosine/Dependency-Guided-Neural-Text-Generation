import torch
import torch.nn as nn
import numpy as np
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from ON_LSTM import ONLSTMStack
from torch.nn import functional as F
import math
from fairseq import utils

class DP_ONLSTMModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, chunk_size, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,tie_weights=False, is_train_dependency=False):
        super(DP_ONLSTMModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM'], 'RNN type is not supported'
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )
        self.decoder = nn.Linear(ninp, ntoken)

        self.query = nn.Linear(in_features=ninp, out_features=ninp)
        self.value = nn.Linear(in_features=ninp, out_features=ninp)
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.is_train_dependency = is_train_dependency
    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.query.bias.data.fill_(0)
        self.query.weight.data.uniform_(-initrange, initrange)
        self.value.bias.data.fill_(0)
        self.value.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input, hidden, return_h=False, context_length=-1):
        if self.is_train_dependency:
            emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )

            emb = self.lockdrop(emb, self.dropouti)
            raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
            self.distance = distances

            output = self.lockdrop(raw_output, self.dropout)

            output = output.view(output.size(0)*output.size(1), output.size(2))
            if return_h:
                return self.decoder(output), hidden, raw_outputs, outputs
            else:
                return self.decoder(output), hidden
        else:
            emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )

            emb = self.lockdrop(emb, self.dropouti)
            raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
            self.distance = distances
            """
            实际上raw_outputs[-1]就是raw_output，raw_outputs[-1]只不过存了rnn每一层的raw_output
            """
            dep_attn_weight = self._pointer(raw_outputs[-1], raw_output, context_length)
            self.distance = distances

            output = self.lockdrop(raw_output, self.dropout)
            result = output.view(output.size(0)*output.size(1), output.size(2))
            if return_h:
                return self.decoder(output), dep_attn_weight, hidden, raw_outputs, outputs
            else:
                return self.decoder(output), dep_attn_weight, hidden

    def generate_step(self, input, hidden):
        emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )
        emb = self.lockdrop(emb, self.dropouti)
        raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
        self.distance = distances
        return raw_output, hidden
            
    def _pointer(self, H, last, context_length):
        """ The `pointer` part of the network

        Parameters
        ----------
        H : Tensor
            Hidden representations for each timestep extracted from the last layer of LSTM
            size : (sequence-len, batch-size, hidden-size)
        last : Tensor
            Representations of the whole sequences (Last vectors from H, extracted earlier for efficiency)
            size : (sequence-len, batch-size, hidden-size)
        """
        seq_len, batch_size, hid_size = H.size()
        query = self.query(last).transpose(0,1) # (b, s, h)
        # H.permute(1,2,0) : [b, h, s]
        attention  = torch.bmm(query, self.value(H).permute(1,2,0)) #  ::(b, s, s)
        attention = attention / math.sqrt(hid_size)
        tgt_seq_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1
            ).to(H.device)
        if context_length < 0:
            attn_mask = tgt_seq_mask 
        else:
            context_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([seq_len, seq_len])), -context_length).to(H.device)
            attn_mask = tgt_seq_mask + context_mask
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention += attn_mask
        weigths = F.softmax(attention, dim=-1)  # ::(b, s, s)
        return weigths
    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)


class DP_RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, is_train_dependency=False):
        super(DP_RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)
        
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.query = nn.Linear(in_features=ninp, out_features=ninp)
        self.value = nn.Linear(in_features=ninp, out_features=ninp)
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.is_train_dependency = is_train_dependency
    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.query.bias.data.fill_(0)
        self.query.weight.data.uniform_(-initrange, initrange)
        self.value.bias.data.fill_(0)
        self.value.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, context_length=-1):
        if self.is_train_dependency:
            emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
            emb = self.lockdrop(emb, self.dropouti)
            raw_output = emb
            new_hidden = []
            raw_outputs = []
            outputs = []
            for l, rnn in enumerate(self.rnns):
                current_input = raw_output
                raw_output, new_h = rnn(raw_output, hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.nlayers - 1:
                    #self.hdrop(raw_output)
                    raw_output = self.lockdrop(raw_output, self.dropouth)
                    outputs.append(raw_output)
            hidden = new_hidden

            output = self.lockdrop(raw_output, self.dropout)
            outputs.append(output)

            if return_h:
                return self.decoder(output), hidden, raw_outputs, outputs
            else:
                return self.decoder(output), hidden
        else:
            emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
            #emb = self.idrop(emb)

            emb = self.lockdrop(emb, self.dropouti)

            raw_output = emb
            new_hidden = []
            raw_outputs = []
            outputs = []
            for l, rnn in enumerate(self.rnns):
                current_input = raw_output
                raw_output, new_h = rnn(raw_output, hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.nlayers - 1:
                    #self.hdrop(raw_output)
                    raw_output = self.lockdrop(raw_output, self.dropouth)
                    outputs.append(raw_output)
            hidden = new_hidden

            output = self.lockdrop(raw_output, self.dropout)
            outputs.append(output)

            dep_attn_weight = self._pointer(raw_outputs[-1], raw_output, context_length)

            result = output.view(output.size(0)*output.size(1), output.size(2))
            if return_h:
                return self.decoder(output), dep_attn_weight, hidden, raw_outputs, outputs
            else:
                return self.decoder(output), dep_attn_weight, hidden
    
    def generate_step(self, input, hidden):
        emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        return raw_output, hidden

    def _pointer(self, H, last, context_length):
        """ The `pointer` part of the network

        Parameters
        ----------
        H : Tensor
            Hidden representations for each timestep extracted from the last layer of LSTM
            size : (sequence-len, batch-size, hidden-size)
        last : Tensor
            Representations of the whole sequences (Last vectors from H, extracted earlier for efficiency)
            size : (sequence-len, batch-size, hidden-size)
        """
        seq_len, batch_size, hid_size = H.size()
        query = self.query(last).transpose(0,1) # (b, s, h)
        # H.permute(1,2,0) : [b, h, s]
        attention  = torch.bmm(query, self.value(H).permute(1,2,0)) #  ::(b, s, s)
        attention = attention / math.sqrt(hid_size)
        tgt_seq_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1
            ).to(H.device)
        if context_length < 0:
            attn_mask = tgt_seq_mask 
        else:
            context_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([seq_len, seq_len])), -context_length).to(H.device)
            attn_mask = tgt_seq_mask + context_mask
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention += attn_mask
        weigths = F.softmax(attention, dim=-1)  # ::(b, s, s)
        return weigths
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
