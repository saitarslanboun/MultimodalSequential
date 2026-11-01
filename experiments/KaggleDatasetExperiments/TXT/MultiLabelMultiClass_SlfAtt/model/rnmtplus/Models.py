''' Define the Transformer model '''

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rnmtplus.multihead_attention import *
from rnmtplus.Layers import EncoderLayer, DecoderLayer
from torch.nn import init

import torch
import torch.nn as nn
import numpy as np
import rnmtplus.Constants as Constants
import torchvision.models as models
import torch.nn.functional as F

__author__ = "Hasan Sait Arslan"

class TextEncoder(nn.Module):
    ''' RNMT+ Encoder. '''

    def __init__(self, n_src_vocab, n_layers=6, d_word_vec=1024, d_model=1024, dropout=0.1):
        super(TextEncoder, self).__init__()

        self.d_model = d_model

        self.src_embed_layer = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = []
        for l in range(n_layers):
            if l == 0:
                self.layer_stack.append(EncoderLayer(d_model, d_model))
            else:
                self.layer_stack.append(EncoderLayer(d_model*2, d_model))
        self.layer_stack = nn.ModuleList(self.layer_stack)

        self.dropout = nn.Dropout(dropout)

        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))

        self.project_nn = nn.Linear(d_model * 2, d_model)
        self.project_init = nn.Linear(d_model * 2, d_model)

    def forward(self, src_seq):

        src_embed = self.src_embed_layer(src_seq)
        src_embed = self.dropout(src_embed)
        src_mask = src_seq.clone()
        src_mask = torch.where(src_seq != 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        
        enc_states = src_embed
        for l, rnn in enumerate(self.layer_stack):
            prev_states = enc_states
            prev_states = pack_padded_sequence(prev_states, lengths=src_mask.sum(1), batch_first=True, enforce_sorted=False)
            enc_states = rnn(prev_states)
            enc_states, _ = pad_packed_sequence(enc_states, batch_first=True)
            enc_states = self.dropout(enc_states)
            if l >= 2:
                prev_states, _ = pad_packed_sequence(prev_states, batch_first=True)
                enc_states = self.residual_scaler * (enc_states + prev_states)
            enc_states = F.layer_norm(enc_states, (self.d_model * 2,))
        enc_states = self.project_nn(enc_states)
        enc_states = F.layer_norm(enc_states, (self.d_model,))
        
        return enc_states         

class RNMTPlus(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=1024, d_model=1024,
            dropout=0.1):

        super(RNMTPlus, self).__init__()
        self.text_encoder = TextEncoder(
            n_src_vocab, n_layers=n_layers,
            d_word_vec=d_word_vec, d_model=d_model,
            dropout=dropout)
        self.tgt_word_projs1 = nn.ModuleDict()
        for a in range(len(n_tgt_vocab)):
            self.tgt_word_projs1[str(a)] = nn.Linear(d_model, d_model)
        self.tgt_word_projs2 = nn.ModuleDict()
        for a in range(len(n_tgt_vocab)):
            self.tgt_word_projs2[str(a)] = nn.Linear(d_model*2, n_tgt_vocab[a], bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab

        self.attention = MultiHeadAttention(d_model, 8)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        params = list(p for p in self.parameters())
        return (p for p in params)

    def forward(self, src):
        src_seq, src_pos = src

        T = self.text_encoder(src_seq)
        T_avg = F.avg_pool1d(T.transpose(1, 2), T.shape[1]).squeeze(2)
        seq_logits = []
        for a in range(len(self.n_tgt_vocab)):
            coutput = self.tgt_word_projs1[str(a)](T_avg)
            seq_logits.append(coutput)
        seq_logits = torch.stack(seq_logits, dim=1)
        comb = torch.cat((seq_logits, T), dim=1)
        context, _ = self.attention(seq_logits, comb, comb)
        att_input = torch.cat((seq_logits, context), dim=2)
        lseq_logits = []
        for a in range(len(self.n_tgt_vocab)):
            coutput = self.tgt_word_projs2[str(a)](att_input[:, a, :])
            lseq_logits.append(coutput)
        return lseq_logits
