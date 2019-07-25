
import torch
import torch.nn as nn
from torch.autograd import Variable

from rnmtplus.Models import RNMTPlus

class Translator(object):

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

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        image_data, src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        beam_size = 1

        V = self.model.image_encoder(image_data)
        encoder_outputs = self.model.text_encoder(src_seq)

        tgt_seq = torch.ones((src_seq.size(0), 1)).long().cuda()

        for a in range(8):
            output, last_hidden, (txt_attention, img_attention) = self.model.decoder(tgt_seq, encoder_outputs, V)
            torch.save(txt_attention, "txt_attention.pt")
            torch.save(img_attention, "img_attention.pt")
            output = self.model.tgt_word_proj(output)
            output = output.argmax(dim=2)
            tgt_seq = torch.cat((tgt_seq, output[:, -1].unsqueeze(1)), dim=1)

        tgt_seq = tgt_seq.tolist()[0] 
        if (len(tgt_seq) == 9) and (tgt_seq[0] == 1) and (tgt_seq[-1] == 2) and (tgt_seq.count(2) == 1):
            return tgt_seq
        else:
            return None

        return seq
