''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from rnmtplus.Models import RNMTPlus

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        if torch.cuda.is_available():
            checkpoint = torch.load(opt.model)
        else:
            checkpoint = torch.load(opt.model, map_location='cpu')
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = RNMTPlus(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        prob_projection = nn.LogSoftmax()

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda()
            prob_projection.cuda()
        else:
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def translate_batch(self, src_batch, vocab_sizes):
        ''' Translation work in one batch '''

        src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        T = self.model.text_encoder(src_seq)
        T = F.avg_pool1d(T.transpose(1, 2), T.shape[1]).squeeze(2)
        seq_logit = self.model.tgt_word_proj(T)
        seq_logit = F.sigmoid(seq_logit)
        values = []
        bind = 0
        for a in range(len(vocab_sizes)):
            ind = bind + vocab_sizes[a]
            ctensor = seq_logit[:, bind:ind]
            values.append(ctensor.argmax(dim=1).tolist())
            bind = ind
        values = torch.LongTensor(values).transpose(0, 1).tolist()        
        return values
