''' The models of the paper: the Sequential model and the baselines.

All models encode the image with ResNet-50 and the title with the RNMT+ encoder (Section 3.1).

- sequential: the Sequential model (Section 3), an RNMT+ decoder that generates the attribute chain.
- The baselines (Section 4.3) average-pool and concatenate the encodings and predict
  - multiclass: one class per attribute chain seen in training (MultiClass),
  - multilabel: one sigmoid per attribute value (Binary MultiLabel),
  - multitask: one softmax classifier per attribute (MultiTask),
  - multitask_att: MultiTask with attention over the image and text features (Attentive MultiTask, Eq. 7),
  - multitask_slfatt: Attentive MultiTask that also attends to the attribute representations
    (Self-Attentive MultiTask, Eq. 8).

The parameter names are those of the original experiment code, so the checkpoints of the paper load.
'''

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rnmtplus.Constants as Constants
from rnmtplus import compat
from rnmtplus.Layers import EncoderLayer, DecoderLayer
from rnmtplus.multihead_attention import MultiHeadAttention

MODEL_TYPES = ('multiclass', 'multilabel', 'multitask', 'multitask_att', 'multitask_slfatt', 'sequential')
MODALITIES = ('txt', 'img', 'txt+img')


def uses_text(modality):
    return modality in ('txt', 'txt+img')


def uses_image(modality):
    return modality in ('img', 'txt+img')


def avg_pool(states):
    ''' The average over all positions; padding positions of a batch are included, as in the original code. '''
    return F.avg_pool1d(states.transpose(1, 2), states.shape[1]).squeeze(2)


class ImageEncoder(nn.Module):
    ''' ResNet-50 up to layer3: a 14x14 grid of 1024-dimensional features for a 224x224 image. '''

    def __init__(self, d_model, pretrained=True):
        super(ImageEncoder, self).__init__()
        resnet = compat.resnet50(pretrained)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-3])

    def forward(self, images):
        A = self.resnet_conv(images)
        return A.view(A.size(0), A.size(1), -1).transpose(1, 2)


class TextEncoder(nn.Module):
    ''' RNMT+ encoder: bidirectional GRU layers, with residual connections from the third layer on. '''

    def __init__(self, n_src_vocab, n_layers=6, d_word_vec=1024, d_model=1024, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.d_model = d_model
        self.src_embed_layer = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model if l == 0 else d_model * 2, d_model) for l in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))
        self.project_nn = nn.Linear(d_model * 2, d_model)
        self.project_init = nn.Linear(d_model * 2, d_model)  # not used; part of the checkpoints

    def forward(self, src_seq):
        ''' The encoder states (batch x length x d_model) and the mask of the tokens (1) and padding (0). '''
        enc_states = self.dropout(self.src_embed_layer(src_seq))
        src_mask = src_seq.ne(Constants.PAD).long()
        lengths = src_mask.sum(1).cpu()
        for l, rnn in enumerate(self.layer_stack):
            prev_states = enc_states
            packed = pack_padded_sequence(prev_states, lengths, batch_first=True, enforce_sorted=False)
            enc_states, _ = pad_packed_sequence(rnn(packed), batch_first=True)
            enc_states = self.dropout(enc_states)
            if l >= 2:
                enc_states = self.residual_scaler * (enc_states + prev_states)
            enc_states = F.layer_norm(enc_states, (self.d_model * 2,))
        enc_states = F.layer_norm(self.project_nn(enc_states), (self.d_model,))
        return enc_states, src_mask


class Decoder(nn.Module):
    ''' RNMT+ decoder: GRU layers; the states of the first layer attend to the title and/or the image. '''

    def __init__(self, n_tgt_vocab, n_layers=8, n_head=4, d_word_vec=1024, d_model=1024, dropout=0.1,
                 use_text=True, use_image=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.use_text = use_text
        self.use_image = use_image
        self.tgt_embed_layer = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        d_input = d_model * (1 + int(use_text) + int(use_image))
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model if l == 0 else d_input, d_model) for l in range(n_layers)])
        if use_text:
            self.attention_txt = MultiHeadAttention(d_model, n_head)
        if use_image:
            self.attention_img = MultiHeadAttention(d_model, n_head)
        self.residual_scaler = torch.sqrt(torch.from_numpy(np.array(0.5, dtype="float32")))

    def forward(self, tgt_seq, encoder_outputs, V):
        ''' tgt_seq: the target prefixes; encoder_outputs: (states, mask) of the text encoder; V: the image
        features. Returns the states of the last layer, its last hidden state and the attention weights of
        the first layer over the title and the image (None for a missing input). '''
        dec_states = self.dropout(self.tgt_embed_layer(tgt_seq))
        txt_attention = img_attention = None
        contexts = []
        for l, rnn in enumerate(self.layer_stack):
            if l == 0:
                dec_states, _ = rnn(dec_states, None)
                dec_states = F.layer_norm(dec_states, (self.d_model,))
                if self.use_text:
                    encoder_states, src_mask = encoder_outputs
                    context_txt, txt_attention = self.attention_txt(
                        dec_states, encoder_states, encoder_states, src_mask)
                    contexts.append(context_txt)
                if self.use_image:
                    context_img, img_attention = self.attention_img(dec_states, V, V)
                    contexts.append(context_img)
            else:
                prev_states = dec_states
                dec_states, last_hidden = rnn(torch.cat([prev_states] + contexts, 2), None)
                dec_states = self.dropout(dec_states)
                if l >= 2:
                    dec_states = self.residual_scaler * (dec_states + prev_states)
                dec_states = F.layer_norm(dec_states, (self.d_model,))
        return dec_states, last_hidden.squeeze(0), (txt_attention, img_attention)


class _Model(nn.Module):

    def _freeze_image_encoder(self):
        ''' The pretrained ResNet is not trained; without gradients, backpropagation stops before it. '''
        if hasattr(self, 'image_encoder'):
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def get_trainable_parameters(self):
        ''' All parameters except those of the pretrained ResNet. '''
        frozen = set(map(id, self.image_encoder.parameters())) if hasattr(self, 'image_encoder') else set()
        return (p for p in self.parameters() if id(p) not in frozen)


class SequentialModel(_Model):
    ''' The Sequential model: an RNMT+ decoder that generates the attribute chain one attribute at a time. '''

    def __init__(self, modality, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8, d_word_vec=1024,
                 d_model=1024, dropout=0.1, pretrained=True):
        super(SequentialModel, self).__init__()
        assert d_model == d_word_vec, \
            'To facilitate the residual connections, the dimensions of all module output shall be the same.'
        self.modality = modality
        if uses_image(modality):
            self.image_encoder = ImageEncoder(d_model, pretrained)
        if uses_text(modality):
            self.text_encoder = TextEncoder(
                n_src_vocab, n_layers=n_layers, d_word_vec=d_word_vec, d_model=d_model, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_layers=n_layers + 2, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model,
            dropout=dropout, use_text=uses_text(modality), use_image=uses_image(modality))
        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)  # not used
        self.d_model = d_model
        self._freeze_image_encoder()

    def encode(self, image, src_seq):
        ''' The image features and the text encoder outputs (None for a missing input). '''
        V = self.image_encoder(image) if uses_image(self.modality) else None
        encoder_outputs = self.text_encoder(src_seq) if uses_text(self.modality) else None
        return V, encoder_outputs

    def forward(self, image, src_seq, tgt_seq):
        ''' The scores of the next token at every position of tgt_seq (<BOS> chain <EOS>), flattened. '''
        V, encoder_outputs = self.encode(image, src_seq)
        dec_states, _, _ = self.decoder(tgt_seq[:, :-1], encoder_outputs, V)
        seq_logit = self.tgt_word_proj(dec_states)
        return seq_logit.view(-1, seq_logit.size(2))


class AttributeModel(_Model):
    ''' The baselines: classifiers on the pooled image and text encodings. '''

    def __init__(self, model_type, modality, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8, d_word_vec=1024,
                 d_model=1024, dropout=0.1, proj1_bias=True, unused_slf_attention=False, pretrained=True):
        ''' n_tgt_vocab: the number of classes (multiclass), of attribute values (multilabel), or the list of
        the vocabulary sizes of the attributes (multitask models). proj1_bias: whether the first projection
        of the attentive models has a bias (not in the Rakuten text+image experiments).
        unused_slf_attention: an attention module that the Rakuten text+image Self-Attentive MultiTask had
        but did not use; needed to load its checkpoint. '''
        super(AttributeModel, self).__init__()
        assert model_type in MODEL_TYPES and model_type != 'sequential'
        self.model_type = model_type
        self.modality = modality
        if uses_image(modality):
            self.image_encoder = ImageEncoder(d_model, pretrained)
        if uses_text(modality):
            self.text_encoder = TextEncoder(
                n_src_vocab, n_layers=n_layers, d_word_vec=d_word_vec, d_model=d_model, dropout=dropout)
        d_features = d_model * (int(uses_image(modality)) + int(uses_text(modality)))
        if model_type in ('multiclass', 'multilabel'):
            self.tgt_word_proj = nn.Linear(d_features, n_tgt_vocab, bias=False)
        elif model_type == 'multitask':
            self.tgt_word_projs = nn.ModuleDict()
            for a in range(len(n_tgt_vocab)):
                self.tgt_word_projs[str(a)] = nn.Linear(d_features, n_tgt_vocab[a], bias=False)
        else:
            self.tgt_word_projs1 = nn.ModuleDict()
            for a in range(len(n_tgt_vocab)):
                self.tgt_word_projs1[str(a)] = nn.Linear(d_features, d_model, bias=proj1_bias)
            self.tgt_word_projs2 = nn.ModuleDict()
            for a in range(len(n_tgt_vocab)):
                self.tgt_word_projs2[str(a)] = nn.Linear(d_model * 2, n_tgt_vocab[a], bias=False)
        self.dropout = nn.Dropout(dropout)  # not used
        if unused_slf_attention:
            self.slf_attention = MultiHeadAttention(d_model, n_head)
        if model_type in ('multitask_att', 'multitask_slfatt'):
            self.attention = MultiHeadAttention(d_model, n_head)
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab
        self._freeze_image_encoder()

    def forward(self, image, src_seq):
        ''' The scores: one tensor (multiclass, multilabel) or one tensor per attribute (multitask models). '''
        V = self.image_encoder(image) if uses_image(self.modality) else None
        T = self.text_encoder(src_seq)[0] if uses_text(self.modality) else None
        encodings = [X for X in (V, T) if X is not None]
        pooled = [avg_pool(X) for X in encodings]
        features = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=1)
        if self.model_type in ('multiclass', 'multilabel'):
            return self.tgt_word_proj(features)
        n_attributes = len(self.n_tgt_vocab)
        if self.model_type == 'multitask':
            return [self.tgt_word_projs[str(a)](features) for a in range(n_attributes)]
        queries = torch.stack([self.tgt_word_projs1[str(a)](features) for a in range(n_attributes)], dim=1)
        if self.model_type == 'multitask_att':
            memory = encodings
        elif self.modality == 'txt+img':  # the order of the original code
            memory = [V, T, queries]
        else:
            memory = [queries] + encodings
        memory = memory[0] if len(memory) == 1 else torch.cat(memory, dim=1)
        context, _ = self.attention(queries, memory, memory)
        att_input = torch.cat((queries, context), dim=2)
        return [self.tgt_word_projs2[str(a)](att_input[:, a, :]) for a in range(n_attributes)]


def build_model(opt, pretrained=True):
    ''' The model described by the options of train.py (also stored as 'settings' in the checkpoints).
    pretrained: load the ImageNet weights into the ResNet (not needed when a checkpoint is loaded). '''
    kwargs = dict(n_layers=getattr(opt, 'n_layers', 6), n_head=getattr(opt, 'n_head', 8),
                  d_word_vec=getattr(opt, 'd_word_vec', opt.d_model), d_model=opt.d_model,
                  dropout=opt.dropout, pretrained=pretrained)
    n_src_vocab = getattr(opt, 'src_vocab_size', None)
    if opt.model_type == 'sequential':
        return SequentialModel(opt.modality, n_src_vocab, opt.tgt_vocab_size, **kwargs)
    return AttributeModel(opt.model_type, opt.modality, n_src_vocab, opt.tgt_vocab_size,
                          proj1_bias=getattr(opt, 'proj1_bias', True),
                          unused_slf_attention=getattr(opt, 'unused_slf_attention', False), **kwargs)


def architecture_from_state_dict(state_dict):
    ''' What the parameter names of a checkpoint tell about the model: the modality, proj1_bias,
    unused_slf_attention, and the model type if it is unambiguous (else None). '''
    keys = list(state_dict.keys())
    has = lambda prefix: any(k.startswith(prefix) for k in keys)
    image, text = has('image_encoder.'), has('text_encoder.')
    modality = 'txt+img' if image and text else ('img' if image else 'txt')
    if has('decoder.'):
        model_type = 'sequential'
    elif has('tgt_word_projs.'):
        model_type = 'multitask'
    else:
        model_type = None  # multiclass or multilabel; multitask_att or multitask_slfatt
    return {'modality': modality, 'model_type': model_type,
            'proj1_bias': 'tgt_word_projs1.0.bias' in keys,
            'unused_slf_attention': has('slf_attention.')}
