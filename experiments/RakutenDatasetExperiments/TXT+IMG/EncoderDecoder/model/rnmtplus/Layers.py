''' Define the Layers '''
import torch.nn as nn
import torch

__author__ = "Hasan Sait Arslan"

class EncoderLayer(nn.Module):

    def __init__(self, d_embed, d_hidden):
        super(EncoderLayer, self).__init__()
        self.encoder_gru = nn.GRU(d_embed, d_hidden, batch_first=True, bidirectional=True)

    def forward(self, enc_input):
        enc_states, _ = self.encoder_gru(enc_input)
        return enc_states

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_embed, d_hidden):
        super(DecoderLayer, self).__init__()
        self.decoder_gru = nn.GRU(d_embed, d_hidden, batch_first=True)

    def forward(self, dec_input, dec_init_state):
        dec_states, dec_last_hidden = self.decoder_gru(dec_input, dec_init_state)
        return dec_states, dec_last_hidden
