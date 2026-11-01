'''
This script handling the training process.
'''

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import rnmtplus.Constants as Constants
from DataLoader import CROPS, DataLoader, image_transform
from preprocess import TGT_FORMATS
from rnmtplus import compat
from rnmtplus.Models import MODALITIES, MODEL_TYPES, build_model, uses_image, uses_text
from rnmtplus.Optim import ScheduledOptim


def get_performance(crit, pred, gold):
    ''' The loss and the number of correct predictions for the non-PAD targets. '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()
    return loss, n_correct


def compute_loss(crit, pred, tgt, opt):
    ''' The loss, the number of correct predictions and the number of targets of a batch (the multilabel model
    only has a loss). For the classifiers, index 0 (<UNK>) has the role of PAD: its targets do not count. '''
    if opt.model_type == 'multilabel':
        gold = torch.cat([torch.zeros((tgt.shape[0], size), device=tgt.device).scatter_(1, tgt[:, a].unsqueeze(1), 1)
                          for a, size in enumerate(opt.tgt_vocab_sizes)], dim=1)
        return crit(pred, gold), None, None
    if opt.model_type in ('sequential', 'multiclass'):
        gold = tgt[:, 1:] if opt.model_type == 'sequential' else tgt
        loss, n_correct = get_performance(crit, pred, gold)
        return loss, n_correct, gold.data.ne(Constants.PAD).sum()
    loss = 0.0
    n_correct = 0.0
    n_words = 0.0
    for a in range(tgt.shape[1]):
        closs, cn_correct = get_performance(crit[a], pred[a], tgt[:, a])
        loss += closs
        n_correct += cn_correct
        n_words += tgt[:, a].data.ne(Constants.PAD).sum()
    return loss, n_correct, n_words


def model_forward(model, image, src, tgt, opt):
    if opt.model_type == 'sequential':
        return model(image, src, tgt)
    return model(image, src)


def epoch_result(total_loss, n_total_words, n_total_correct, opt):
    ''' (loss per target, accuracy); (summed loss, None) for the multilabel model. '''
    if opt.model_type == 'multilabel':
        return total_loss, None
    return total_loss / n_total_words.float(), n_total_correct.float() / n_total_words.float()


def run_epoch(model, data, crit, opt, optimizer=None):
    ''' One epoch of training (with an optimizer) or of evaluation (without). '''
    training = optimizer is not None
    model.train(training)
    total_loss = 0
    n_total_words = 0
    n_total_correct = 0
    with torch.set_grad_enabled(training):
        t = tqdm(data, mininterval=1, desc='Training' if training else 'Validating', leave=False)
        for _, image, src, tgt in t:
            if training:
                optimizer.zero_grad()
            pred = model_forward(model, image, src, tgt, opt)
            loss, n_correct, n_words = compute_loss(crit, pred, tgt, opt)
            if training:
                loss.backward()
                optimizer.step()
                optimizer.update_learning_rate()
            if n_words is not None:
                n_total_words += n_words
                n_total_correct += n_correct
            loss_value = loss.item()
            total_loss += loss_value
            t.set_description("Loss: " + str(loss_value))
    return epoch_result(total_loss, n_total_words, n_total_correct, opt)


def train_epoch(model, training_data, crit, optimizer, opt):
    ''' Epoch operation in training phase'''
    return run_epoch(model, training_data, crit, opt, optimizer)


def eval_epoch(model, validation_data, crit, opt):
    ''' Epoch operation in evaluation phase '''
    return run_epoch(model, validation_data, crit, opt)


def performance_line(name, loss, accu, start, opt):
    if opt.model_type == 'multilabel':
        return '  - ({name})   loss: {loss: 8.5f} %, elapse: {elapse:3.3f} min'.format(
            name=name, loss=loss, elapse=(time.time() - start) / 60)
    return '  - ({name})   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
        name=name, ppl=math.exp(min(loss, 100)), accu=100 * accu, elapse=(time.time() - start) / 60)


def log_line(epoch, loss, accu, opt):
    if opt.model_type == 'multilabel':
        return '{epoch},{loss: 8.5f}\n'.format(epoch=epoch, loss=loss)
    return '{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
        epoch=epoch, loss=loss, ppl=math.exp(min(loss, 100)), accu=100 * accu)


def train(model, training_data, validation_data, crit, optimizer, opt):
    ''' Start training '''
    log_train_file = None
    log_valid_file = None
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        print('[Info] Training performance will be written to file: {}'.format(log_train_file))
        print('[Info] Validation performance will be written to file: {}'.format(log_valid_file))
        header = 'epoch,loss\n' if opt.model_type == 'multilabel' else 'epoch,loss,ppl,accuracy\n'
        for log_file in (log_train_file, log_valid_file):
            with open(log_file, 'w') as f:
                f.write(header)
    if opt.save_model and not os.path.isdir(opt.save_model):
        os.makedirs(opt.save_model)

    best = None  # the best validation accuracy; the lowest validation loss for the multilabel model
    epochs_without_improvement = 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer, opt)
        print(performance_line('Training', train_loss, train_accu, start, opt))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit, opt)
        print(performance_line('Validation', valid_loss, valid_accu, start, opt))

        score = -float(valid_loss) if opt.model_type == 'multilabel' else float(valid_accu)
        improved = best is None or score >= best
        if improved:
            best = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {'model': model.state_dict(), 'settings': opt, 'epoch': epoch_i}
        if opt.save_model:
            if opt.save_mode == 'all':
                torch.save(checkpoint, os.path.join(opt.save_model, "model." + str(epoch_i) + ".chkpt"))
            elif improved:
                torch.save(checkpoint, os.path.join(opt.save_model, 'model.best.chkpt'))
                print('    - [Info] The checkpoint file has been updated.')

        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                log_tf.write(log_line(epoch_i, train_loss, train_accu, opt))
            with open(log_valid_file, 'a') as log_vf:
                log_vf.write(log_line(epoch_i, valid_loss, valid_accu, opt))

        if opt.patience and epochs_without_improvement >= opt.patience:
            print('[Info] No improvement on the validation set for {} epochs; stopping.'.format(opt.patience))
            break


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES)
    parser.add_argument('-modality', choices=MODALITIES, help='default: the inputs in -data')
    parser.add_argument('-data', required=True, help='data.pt (preprocess.py)')
    parser.add_argument('-image_dir', help='the image directory (image models)')

    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('-train_crop', choices=sorted(CROPS), default='random_crop',
                        help='the random crop of the images in training and validation')
    parser.add_argument('-no_proj1_bias', action='store_true',
                        help='attentive models: no bias in the first projection (Rakuten text+image)')
    parser.add_argument('-unused_slf_attention', action='store_true',
                        help='add the attention module that the Rakuten text+image Self-Attentive '
                             'MultiTask model had but did not use')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None, help='the directory for the checkpoints')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best',
                        help='all: models/model.<epoch>.chkpt after every epoch; best: models/model.best.chkpt '
                             'whenever the validation accuracy (multilabel: loss) is the best so far')
    parser.add_argument('-patience', type=int, default=0,
                        help='stop when the validation result has not improved for this many epochs '
                             '(0: train for -epoch epochs)')
    parser.add_argument('-pretrained', type=str, default="", help='a checkpoint to continue from')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-num_workers', type=int, default=4, help='processes that load the images')
    parser.add_argument('-seed', type=int, default=None, help='seed of the random number generators')
    parser.add_argument('-deterministic', action='store_true',
                        help='deterministic cuDNN algorithms (slower) instead of the fastest ones')
    parser.add_argument('-tf32', action='store_true',
                        help='allow TensorFloat-32 on Ampere and newer GPUs: faster, but less precise than the '
                             'float32 of the experiments')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.proj1_bias = not opt.no_proj1_bias
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.deterministic
    torch.backends.cudnn.deterministic = opt.deterministic
    compat.set_tf32(opt.tf32)

    #========= Loading Dataset =========#
    data = compat.torch_load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len
    has_image, has_text = 'img' in data['train'], 'src' in data['train']
    if opt.modality is None:
        opt.modality = 'txt+img' if has_image and has_text else ('img' if has_image else 'txt')
    if (uses_image(opt.modality) and not has_image) or (uses_text(opt.modality) and not has_text):
        parser.error('-data has no {} inputs'.format('image' if uses_image(opt.modality) and not has_image
                                                     else 'text'))
    if uses_image(opt.modality) and not opt.image_dir:
        parser.error('the image models need -image_dir')
    tgt_format = TGT_FORMATS[opt.model_type]
    tgt_dict = data['dict']['tgt']

    #========= Preparing DataLoader =========#
    transform = image_transform(opt.train_crop, opt.crop_size) if uses_image(opt.modality) else None

    def loader(split, batch_size, shuffle):
        return DataLoader(
            transform, opt.image_dir, data['dict'].get('src'), tgt_dict,
            image_insts=data[split]['img'] if uses_image(opt.modality) else None,
            src_insts=data[split]['src'] if uses_text(opt.modality) else None,
            tgt_insts=data[split]['tgt'], tgt_format=tgt_format, batch_size=batch_size, shuffle=shuffle,
            cuda=opt.cuda, num_workers=opt.num_workers)

    training_data = loader('train', opt.batch_size, True)
    validation_data = loader('valid', int(opt.batch_size / 3), False)

    if uses_text(opt.modality):
        opt.src_vocab_size = len(data['dict']['src'])
    if tgt_format == 'attributes':
        sizes = [len(tgt_dict[str(a)]) for a in range(len(tgt_dict.keys()))]
        if opt.model_type == 'multilabel':
            opt.tgt_vocab_size, opt.tgt_vocab_sizes = sum(sizes), sizes
        else:
            opt.tgt_vocab_size = sizes
    else:
        opt.tgt_vocab_size = len(tgt_dict)

    print(opt)

    #========= Preparing Model =========#
    model = build_model(opt)

    if opt.pretrained:
        checkpoint = compat.torch_load(opt.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        num_of_steps = int(opt.pretrained.split(".")[1])
    else:
        num_of_steps = 0

    optimizer = ScheduledOptim(
        optim.Adam(model.get_trainable_parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, num_of_steps)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, reduction='sum')

    if opt.model_type == 'multilabel':
        crit = nn.BCEWithLogitsLoss()
    elif tgt_format == 'attributes':
        crit = [get_criterion(size) for size in opt.tgt_vocab_size]
    else:
        crit = get_criterion(opt.tgt_vocab_size)

    if opt.cuda:
        model = model.cuda()
        crit = [c.cuda() for c in crit] if isinstance(crit, list) else crit.cuda()

    train(model, training_data, validation_data, crit, optimizer, opt)


if __name__ == '__main__':
    main()
