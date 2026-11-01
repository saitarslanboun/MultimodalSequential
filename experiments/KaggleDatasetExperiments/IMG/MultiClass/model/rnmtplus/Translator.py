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
            model_opt.tgt_vocab_size,
            d_model=model_opt.d_model,
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

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        image_data = src_batch
        batch_size = image_data.size(0)

        V = self.model.image_encoder(image_data)
        V = F.avg_pool1d(V.transpose(1, 2), V.shape[1]).squeeze(2)
        seq_logit = self.model.tgt_word_proj(V)        
        outputs = seq_logit.argmax(dim=1)
 
        return outputs
