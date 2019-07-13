'''
This script handling the training process.
'''

from tqdm import tqdm
from rnmtplus.Models import RNMTPlus
from rnmtplus.Optim import ScheduledOptim
from DataLoader import DataLoader
from torchvision import transforms

import argparse
import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import rnmtplus.Constants as Constants

def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

    return loss, n_correct

def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    t = tqdm(training_data, mininterval=1, desc='Training', leave=False)
    for batch in t:

        # prepare data
        image, src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(image, src, tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.item()

        description = "Loss: " + str(loss.item())
        t.set_description(description)

    return total_loss/n_total_words.float(), n_total_correct.float()/n_total_words.float()

def eval_epoch(model, validation_data, crit, opt, epoch):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    t = tqdm(validation_data, mininterval=1, desc='Validating', leave=False)
    for batch in t:

        # prepare data
        image, src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(image, src, tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold) 

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.item()

        description = "Loss: " + str(loss.item())
        t.set_description(description)

    return total_loss/n_total_words.float(), n_total_correct.float()/n_total_words.float()

def train(model, training_data, validation_data, crit, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + ".valid.log"

        print('[Info] Training performance will be written to file: {}'.format(
            log_train_file))
        print('[Info] Validation performance will be written to file: {}'.format(
            log_valid_file))

        with open(log_train_file, 'w') as log_tf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
        with open(log_valid_file, 'w') as log_vf:
            log_vf.write('epoch,loss,ppl,accuracy\n')

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit, opt, epoch_i)
        print('  - (Validation)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                  elapse=(time.time()-start)/60))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            model_name = os.path.join(opt.save_model, "model." + str(epoch_i) + ".chkpt")
            torch.save(checkpoint, model_name)

        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
            with open(log_valid_file, 'a') as log_vf:
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-image_dir', required=True)

    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-validation_output', default='validations')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('-pretrained', type=str, default="")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    #========= Image Preprocessing =========#
    transform = transforms.Compose([
        transforms.RandomCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    #========= Preparing DataLoader =========#
    training_data = DataLoader(
        transform,
        opt.image_dir,
        data['dict']['src'],
        data['dict']['tgt'],
        image_insts=data['train']['img'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    validation_data = DataLoader(
        transform,
        opt.image_dir,
        data['dict']['src'],
        data['dict']['tgt'],
        image_insts=data['valid']['img'],
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=int(opt.batch_size/3),
        shuffle=False,
        test=True,
        cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_size = training_data.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight and training_data.src_word2idx != training_data.tgt_word2idx:
        print('[Warning]',
              'The src/tgt word2idx table are different but asked to share word embedding.')

    print(opt)

    rnmtplus = RNMTPlus(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    if opt.pretrained:
        checkpoint = torch.load(opt.pretrained)
        rnmtplus.load_state_dict(checkpoint['model'])
        num_of_steps = int(opt.pretrained.split(".")[1])
    else:
        num_of_steps = 0

    optimizer = ScheduledOptim(
        optim.Adam(
            rnmtplus.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, num_of_steps)


    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(training_data.tgt_vocab_size)

    if opt.cuda:
        rnmtplus = rnmtplus.cuda()
        crit = crit.cuda()

    train(rnmtplus, training_data, validation_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
