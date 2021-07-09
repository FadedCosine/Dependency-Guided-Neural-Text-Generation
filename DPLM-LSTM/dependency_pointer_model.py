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

class DPRNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, chunk_size, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, is_train_dependency=False):
        super(DPRNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        assert rnn_type in ['LSTM'], 'RNN type is not supported'
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )
        self.decoder = nn.Linear(ninp, ntoken)

        self.query = nn.Linear(in_features=ninp, out_features=ninp)

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
    
    def forward(self, input, hidden, return_h=False):
        if self.is_train_dependency:
            emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )

            emb = self.lockdrop(emb, self.dropouti)
            raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
            self.distance = distances

            output = self.lockdrop(raw_output, self.dropout)

            result = output.view(output.size(0)*output.size(1), output.size(2))
            if return_h:
                return result, hidden, raw_outputs, outputs
            else:
                return result, hidden
        else:
            emb = embedded_dropout(
                self.encoder, input,
                dropout=self.dropoute if self.training else 0
            )

            emb = self.lockdrop(emb, self.dropouti)
            raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
            self.distance = distances
            dep_attn_weight = self._pointer(raw_outputs[-1], raw_output)
            self.distance = distances
            
    def _pointer(self, H, last):
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
        query = F.tanh(self.query(last)).transpose(0,1) # (b, s, h)

        attention  = torch.bmm(query, H.permute(1,2,0)) #  ::(b, s, s)
        attention = attention / math.sqrt(hid_size)
        tgt_seq_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1
            )
        tgt_seq_mask = tgt_seq_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention += tgt_seq_mask
        weigths = F.softmax(attention, dim=1)  # ::(b, s)

        return weigths