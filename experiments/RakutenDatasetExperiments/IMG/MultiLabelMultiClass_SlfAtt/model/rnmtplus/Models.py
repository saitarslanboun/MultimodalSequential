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

class ImageEncoder(nn.Module):
    def __init__(self, d_model):
        super(ImageEncoder, self).__init__()

        # ResNet-50 backend
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-3]
        resnet_conv = nn.Sequential(*modules)

        self.resnet_conv = resnet_conv

    def forward(self, images):
        A = self.resnet_conv(images)
        V = A.view(A.size(0), A.size(1), -1).transpose(1, 2)
        return V

class RNMTPlus(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6,
            d_model=1024,
            dropout=0.1):

        super(RNMTPlus, self).__init__()
        self.image_encoder = ImageEncoder(d_model)
        self.tgt_word_projs1 = nn.ModuleDict()
        for a in range(len(n_tgt_vocab)):
            coutput = self.tgt_word_projs1[str(a)] = nn.Linear(d_model, d_model)
        self.tgt_word_projs2 = nn.ModuleDict()
        for a in range(len(n_tgt_vocab)):
            self.tgt_word_projs2[str(a)] = nn.Linear(d_model*2, n_tgt_vocab[a], bias=False)
        self.attention = MultiHeadAttention(d_model, 8)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set(map(id, self.image_encoder.parameters()))
        params = list(p for p in self.parameters() if id(p) not in freezed_param_ids)
        return (p for p in params)

    def forward(self, image):
        V = self.image_encoder(image)
        V_avg = F.avg_pool1d(V.transpose(1, 2), V.shape[1]).squeeze(2)
        seq_logits = []
        for a in range(len(self.n_tgt_vocab)):
            coutput = self.tgt_word_projs1[str(a)](V_avg)
            seq_logits.append(coutput)
        seq_logits = torch.stack(seq_logits, dim=1)
        comb = torch.cat((seq_logits, V), dim=1)
        context, _ = self.attention(seq_logits, comb, comb)
        att_input = torch.cat((seq_logits, context), dim=2)
        lseq_logits = []
        for a in range(len(self.n_tgt_vocab)):
            coutput = self.tgt_word_projs2[str(a)](att_input[:, a, :])
            lseq_logits.append(coutput)
        return lseq_logits
