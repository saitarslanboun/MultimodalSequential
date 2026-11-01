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
        self.avgpool = nn.AvgPool2d(14)

    def forward(self, images):
        A = self.resnet_conv(images)
        V = A.view(A.size(0), A.size(1), -1).transpose(1, 2)
        g = self.avgpool(A)
        g = g.view(g.size(0), -1)
        return V

class Decoder(nn.Module):
    ''' A decoder model with multi-head attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_layers=8, n_head=4, d_word_vec=1024, d_model=1024, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model

        self.tgt_embed_layer = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = []
        for l in range(n_layers):
            if l == 0:
                self.layer_stack.append(DecoderLayer(d_model, d_model))
            else:
                self.layer_stack.append(DecoderLayer(d_model*2, d_model))
        self.layer_stack = nn.ModuleList(self.layer_stack)

        self.attention_img = MultiHeadAttention(d_model, n_head)

        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))

    def forward(self, tgt_seq, dec_init_state, V, return_attns=False):

        tgt_embed = self.tgt_embed_layer(tgt_seq)
        tgt_embed = self.dropout(tgt_embed)

        dec_states = None
        for l, rnn in enumerate(self.layer_stack):
            if l == 0:
                dec_states, _ = rnn(tgt_embed, dec_init_state)
                dec_states = F.layer_norm(dec_states, (self.d_model,))
                context_img, _ = self.attention_img(dec_states, V, V)
            else:
                prev_states = dec_states
                dec_input = torch.cat([prev_states, context_img], 2)
                dec_states, last_hidden = rnn(dec_input, dec_init_state)
                dec_states = self.dropout(dec_states)
                if l >= 2:
                    dec_states = self.residual_scaler * (dec_states + prev_states)
                dec_states = F.layer_norm(dec_states, (self.d_model,))
        return dec_states, last_hidden.squeeze(0)
         

class RNMTPlus(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=1024, d_model=1024,
            dropout=0.1):

        super(RNMTPlus, self).__init__()
        self.image_encoder = ImageEncoder(d_model)
        self.decoder = Decoder(
            n_tgt_vocab, n_layers=n_layers+2, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            dropout=dropout)
        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set(map(id, self.image_encoder.parameters()))
        params = list(p for p in self.parameters() if id(p) not in freezed_param_ids)
        return (p for p in params)

    def forward(self, image, tgt):
        tgt_seq, tgt_pos = tgt
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        V = self.image_encoder(image)
        decoder_output, _ = self.decoder(tgt_seq, None, V)
        seq_logit = self.tgt_word_proj(decoder_output)

        return seq_logit.view(-1, seq_logit.size(2))

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
