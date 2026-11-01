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
            self, n_tgt_vocab,
            d_model=1024,
            dropout=0.1):

        super(RNMTPlus, self).__init__()
        self.image_encoder = ImageEncoder(d_model)
        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set(map(id, self.image_encoder.parameters()))
        params = list(p for p in self.parameters() if id(p) not in freezed_param_ids)
        return (p for p in params)

    def forward(self, image):

        V = self.image_encoder(image)
        V = F.avg_pool1d(V.transpose(1, 2), V.shape[1]).squeeze(2)
        seq_logit = self.tgt_word_proj(V)

        return seq_logit

    def validate(self, image, src):

        V = self.image_encoder(image)
        encoder_outputs = self.text_encoder(src)

        tgt_seq = torch.ones((src.size(0), 1)).long().cuda()
        for a in range(49):
            output, last_hidden = self.decoder(tgt_seq, encoder_outputs, V)
            output = self.tgt_word_proj(output)
            output = output.argmax(2)[:, -1]
            tgt_seq = torch.cat((tgt_seq.clone(), output.unsqueeze(1)), dim=1)
            encoder_outputs['dec_init_state'] = last_hidden.unsqueeze(0)
        
        return tgt_seq
