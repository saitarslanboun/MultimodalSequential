''' Prediction with a trained model: the highest-scoring classes for the baselines, greedy decoding for the
Sequential model. '''

import torch

import rnmtplus.Constants as Constants
from rnmtplus import compat
from rnmtplus.Models import architecture_from_state_dict, build_model


class Translator(object):
    ''' Loads a checkpoint (a file models/model.<epoch>.chkpt written by train.py) for prediction. '''

    def __init__(self, opt):
        ''' opt: the options of translate.py. opt.model_type is only used for checkpoints of the original
        experiment code, which do not record the model type. '''
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        checkpoint = compat.torch_load(opt.model, map_location='cpu')
        model_opt = checkpoint['settings']
        architecture = architecture_from_state_dict(checkpoint['model'])
        if getattr(model_opt, 'model_type', None) is None:
            model_opt.model_type = architecture['model_type'] or getattr(opt, 'model_type', None)
            if model_opt.model_type is None:
                raise ValueError('The checkpoint does not record its model type; give it with -model_type.')
        for name in ('modality', 'proj1_bias', 'unused_slf_attention'):
            setattr(model_opt, name, architecture[name])
        self.model_opt = model_opt

        self.model = build_model(model_opt, pretrained=False)
        self.model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        self.model.to(self.device)
        self.model.eval()

    def translate_batch(self, image, src_seq, vocab_sizes=None):
        ''' The predictions for a batch, one per product: a class index (multiclass), a list with the index of
        the value of every attribute (multilabel and the multitask models), or, for the sequential model,
        the decoded token indices (<BOS> first); with opt.check_chain, None unless the sequence ends with
        its only <EOS>. vocab_sizes: the attribute vocabulary sizes (multilabel). For the sequential model,
        also returns the attention weights of the last decoding step (text, image). '''
        model_type = self.model_opt.model_type
        with torch.no_grad():
            if model_type == 'sequential':
                return self._decode(image, src_seq)
            scores = self.model(image, src_seq)
            if model_type == 'multiclass':
                return scores.argmax(dim=1).tolist()
            if model_type == 'multilabel':
                scores = torch.sigmoid(scores)
                values = []
                begin = 0
                for size in vocab_sizes:
                    values.append(scores[:, begin:begin + size].argmax(dim=1).tolist())
                    begin += size
            else:
                values = [attribute_scores.argmax(dim=1).tolist() for attribute_scores in scores]
            return [list(item) for item in zip(*values)]

    def _decode(self, image, src_seq):
        model = self.model
        V, encoder_outputs = model.encode(image, src_seq)
        batch_size = (image if image is not None else src_seq).size(0)
        tgt_seq = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)  # <BOS>
        attentions = (None, None)
        for _ in range(self.opt.decode_steps):
            output, _, attentions = model.decoder(tgt_seq, encoder_outputs, V)
            output = model.tgt_word_proj(output).argmax(dim=2)
            tgt_seq = torch.cat((tgt_seq, output[:, -1].unsqueeze(1)), dim=1)
        seqs = tgt_seq.tolist()
        if self.opt.check_chain:
            seqs = [seq if len(seq) == self.opt.decode_steps + 1 and seq[0] == Constants.BOS
                    and seq[-1] == Constants.EOS and seq.count(Constants.EOS) == 1 else None
                    for seq in seqs]
        return seqs, attentions
