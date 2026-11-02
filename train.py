'''
This script handling the training process.
'''

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import kaggle_data
import rnmtplus.Constants as Constants
from DataLoader import CROPS, DataLoader, SeededRandomTransform, check_image_dir, image_transform
from evaluate import attribute_errors
from preprocess import TGT_FORMATS
from rnmtplus import compat
from rnmtplus.Models import MODALITIES, MODEL_TYPES, build_model, uses_image, uses_text
from rnmtplus.Optim import ScheduledOptim
from rnmtplus.Translator import ChainFormatter, Translator, make_batches


def counted(gold):
    ''' The targets that count: not PAD (index 0) and not a missing value (Constants.MISSING). '''
    return gold.ne(Constants.PAD) & gold.ne(Constants.MISSING)


def smoothed_cross_entropy(pred, gold, eps):
    ''' The cross-entropy with label smoothing (Vaswani et al. 2017), as attention-is-all-you-need-pytorch,
    which the original code is based on, implements it: the gold class gets 1 - eps, every other class
    eps / (classes - 1); summed over the targets other than PAD (index 0) and missing values. '''
    one_hot = torch.zeros_like(pred).scatter(1, gold.clamp(min=0).view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (pred.size(1) - 1)
    loss = -(one_hot * F.log_softmax(pred, dim=1)).sum(dim=1)
    return loss.masked_select(counted(gold)).sum()


def get_performance(crit, pred, gold, smoothing=0.0):
    ''' The loss and the number of correct predictions for the targets other than PAD and missing values. '''
    gold = gold.contiguous().view(-1)
    loss = smoothed_cross_entropy(pred, gold, smoothing) if smoothing else crit(pred, gold)
    pred = pred.max(1)[1]
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(counted(gold).data).sum()
    return loss, n_correct


def compute_loss(crit, pred, tgt, opt):
    ''' The loss, the number of correct predictions and the number of targets of a batch (the multilabel model
    only has a loss). For the classifiers, index 0 (<UNK>) has the role of PAD: its targets do not count.
    Missing values (Constants.MISSING) do not count either: the cross-entropy ignores them (ignore_index), and
    the multilabel model leaves the binary targets of a missing attribute out of its loss.
    With -label_smoothing, the multilabel model smooths the binary targets of each attribute like the
    cross-entropy of the other models: 1 - eps for the gold value, eps / (values - 1) for the others. '''
    eps = getattr(opt, 'label_smoothing', 0.0)
    if opt.model_type == 'multilabel':
        blocks, known = [], []
        for a, size in enumerate(opt.tgt_vocab_sizes):
            values = tgt[:, a]
            block = torch.zeros((tgt.shape[0], size), device=tgt.device).scatter_(1, values.clamp(min=0).unsqueeze(1), 1)
            if eps and size > 1:
                block = block * (1 - eps) + (1 - block) * eps / (size - 1)
            blocks.append(block)
            known.append(values.ne(Constants.MISSING).unsqueeze(1).expand(-1, size))
        gold = torch.cat(blocks, dim=1)
        if not bool((tgt == Constants.MISSING).any()):
            return crit(pred, gold), None, None
        mask = torch.cat(known, dim=1).to(pred.dtype)  # the mean over the binary targets of known attributes
        loss = (F.binary_cross_entropy_with_logits(pred, gold, reduction='none') * mask).sum() / mask.sum()
        return loss, None, None
    if opt.model_type in ('sequential', 'multiclass'):
        gold = tgt[:, 1:] if opt.model_type == 'sequential' else tgt
        loss, n_correct = get_performance(crit, pred, gold, eps)
        return loss, n_correct, counted(gold.data).sum()
    loss = 0.0
    n_correct = 0.0
    n_words = 0.0
    for a in range(tgt.shape[1]):
        closs, cn_correct = get_performance(crit[a], pred[a], tgt[:, a], eps)
        loss += closs
        n_correct += cn_correct
        n_words += counted(tgt[:, a].data).sum()
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


def float_outputs(pred):
    ''' The model outputs in float32, for the loss (mixed precision computes them in float16). '''
    return [p.float() for p in pred] if isinstance(pred, list) else pred.float()


def run_epoch(model, data, crit, opt, optimizer=None, scaler=None):
    ''' One epoch of training (with an optimizer) or of evaluation (without). With -amp, the model computes in
    mixed precision, the loss in float32, and scaler scales the loss so that float16 gradients do not underflow. '''
    training = optimizer is not None
    model.train(training)
    if training and opt.frozen_resnet and hasattr(model, 'image_encoder'):
        model.image_encoder.eval()  # BatchNorm uses and keeps the ImageNet statistics
    total_loss = 0
    n_total_words = 0
    n_total_correct = 0
    with torch.set_grad_enabled(training):
        t = tqdm(data, mininterval=1, desc='Training' if training else 'Validating', leave=False)
        for _, image, src, tgt in t:
            if training:
                optimizer.zero_grad()
            with compat.autocast(opt.amp):
                pred = model_forward(model, image, src, tgt, opt)
            if opt.amp:
                pred = float_outputs(pred)
            loss, n_correct, n_words = compute_loss(crit, pred, tgt, opt)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer.optimizer)
                    scaler.update()
                else:
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


def train_epoch(model, training_data, crit, optimizer, opt, scaler=None):
    ''' Epoch operation in training phase'''
    return run_epoch(model, training_data, crit, opt, optimizer, scaler)


def eval_epoch(model, validation_data, crit, opt):
    ''' Epoch operation in evaluation phase '''
    return run_epoch(model, validation_data, crit, opt)


def performance_line(name, loss, accu, start, opt):
    if opt.model_type == 'multilabel':
        return '  - ({name})   loss: {loss: 8.5f} %, elapse: {elapse:3.3f} min'.format(
            name=name, loss=loss, elapse=(time.time() - start) / 60)
    return '  - ({name})   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
        name=name, ppl=math.exp(min(loss, 100)), accu=100 * accu, elapse=(time.time() - start) / 60)


def log_line(epoch, loss, accu, opt, chain_accuracy=None):
    if opt.model_type == 'multilabel':
        line = '{epoch},{loss: 8.5f}'.format(epoch=epoch, loss=loss)
    else:
        line = '{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}'.format(
            epoch=epoch, loss=loss, ppl=math.exp(min(loss, 100)), accu=100 * accu)
    if chain_accuracy is not None:
        line += ',{:3.3f}'.format(100 * chain_accuracy)
    return line + '\n'


class ChainAccuracy(object):
    ''' The full-chain accuracy of the model's predictions for the validation set (-select_by chain_accuracy):
    the classifiers' highest-scoring values, the greedy decoding of the sequential model, as translate.py
    predicts, counted as evaluate.py counts errors. '''

    def __init__(self, model, data, transform, opt, device):
        modality = opt.modality
        src_insts = data['valid']['src'] if uses_text(modality) else None
        img_insts = data['valid']['img'] if uses_image(modality) else None
        self.gold = data['valid']['chains']
        self.loader = DataLoader(
            transform, opt.image_dir, data['dict'].get('src'), data['dict']['tgt'], image_insts=img_insts,
            src_insts=src_insts, cuda=opt.cuda, shuffle=False, num_workers=opt.num_workers,
            seed=opt.seed if opt.kaggle_dir else None,
            batches=make_batches(opt.model_type, src_insts, len(self.gold), opt.batch_size))
        check_chain = not opt.no_check_chain
        self.translator = Translator.from_model(
            model, opt, argparse.Namespace(decode_steps=opt.decode_steps, check_chain=check_chain, amp=opt.amp),
            device)
        self.chain = ChainFormatter(opt.model_type, data['dict']['tgt'], check_chain)
        self.model, self.opt = model, opt
        self.n_attributes = max(len(chain.split()) for chain in self.gold)
        missing = getattr(data['settings'], 'missing_values', '')
        self.missing = frozenset(v for v in missing.split(',') if v)

    def __call__(self):
        self.model.eval()
        lines = [None] * len(self.gold)
        for indices, image, src, _ in tqdm(self.loader, mininterval=1, desc='Predicting', leave=False):
            if self.opt.model_type == 'sequential':
                predictions, _ = self.translator.translate_batch(image, src)
            else:
                predictions = self.translator.translate_batch(image, src, getattr(self.opt, 'tgt_vocab_sizes', None))
            for i, prediction in zip(indices, predictions):
                lines[i] = self.chain(prediction)
        _, errors_all = attribute_errors(self.gold, lines, self.n_attributes, self.missing)
        return 1.0 - float(errors_all) / len(self.gold)


def train(model, training_data, validation_data, crit, optimizer, opt, chain_accuracy=None, scaler=None):
    ''' Start training '''
    log_train_file = None
    log_valid_file = None
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        print('[Info] Training performance will be written to file: {}'.format(log_train_file))
        print('[Info] Validation performance will be written to file: {}'.format(log_valid_file))
        header = 'epoch,loss' if opt.model_type == 'multilabel' else 'epoch,loss,ppl,accuracy'
        with open(log_train_file, 'w') as f:
            f.write(header + '\n')
        with open(log_valid_file, 'w') as f:
            f.write(header + (',chain_accuracy' if chain_accuracy else '') + '\n')
    if opt.save_model and not os.path.isdir(opt.save_model):
        os.makedirs(opt.save_model)

    best = None  # the best validation result (see -select_by)
    epochs_without_improvement = 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer, opt, scaler)
        print(performance_line('Training', train_loss, train_accu, start, opt))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit, opt)
        print(performance_line('Validation', valid_loss, valid_accu, start, opt))
        valid_chain_accuracy = None
        if chain_accuracy:
            start = time.time()
            valid_chain_accuracy = chain_accuracy()
            print('  - (Validation)   full-chain accuracy of the predictions: {:3.3f} %, elapse: {:3.3f} min'.format(
                100 * valid_chain_accuracy, (time.time() - start) / 60))

        if opt.select_by == 'chain_accuracy':
            score = valid_chain_accuracy
        else:
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
            elif opt.save_mode == 'last':
                torch.save(checkpoint, os.path.join(opt.save_model, 'model.last.chkpt'))
            elif improved:
                torch.save(checkpoint, os.path.join(opt.save_model, 'model.best.chkpt'))
                print('    - [Info] The checkpoint file has been updated.')

        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                log_tf.write(log_line(epoch_i, train_loss, train_accu, opt))
            with open(log_valid_file, 'a') as log_vf:
                log_vf.write(log_line(epoch_i, valid_loss, valid_accu, opt, valid_chain_accuracy))

        if opt.patience and epochs_without_improvement >= opt.patience:
            print('[Info] No improvement on the validation set for {} epochs; stopping.'.format(opt.patience))
            break


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES)
    parser.add_argument('-modality', choices=MODALITIES, help='default: the inputs in -data')
    parser.add_argument('-data', help='data.pt (preprocess.py); or -kaggle_dir')
    parser.add_argument('-image_dir',
                        help='the image directory (image models; default with -kaggle_dir: its images/), or the '
                             'images resized in advance by resize_images.py, for the crops that resize first '
                             '(center_crop, resize_random_crop): the same images, read faster')

    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('-train_crop', choices=sorted(CROPS),
                        help='the crop of the images in training (default: random_crop; with -kaggle_dir '
                             'random_resized_crop, as in the paper\'s Kaggle runs). resize_random_crop: resize '
                             'the shorter side to 256 (for -crop_size 224) and crop at random, which keeps '
                             'most of the image, unlike random_resized_crop (8%% to 100%% of it)')
    parser.add_argument('-valid_crop', choices=sorted(CROPS),
                        help='the crop of the images in validation (default: -train_crop, as in the paper\'s '
                             'runs; with -kaggle_dir center_crop)')
    parser.add_argument('-no_proj1_bias', action='store_true',
                        help='attentive models: no bias in the first projection (Rakuten text+image)')
    parser.add_argument('-unused_slf_attention', action='store_true',
                        help='add the attention module that the Rakuten text+image Self-Attentive '
                             'MultiTask model had but did not use')
    parser.add_argument('-mask_padding', action='store_true',
                        help='the baselines leave out the padding of the titles in the average pooling and in '
                             'the attention, as the attention of the sequential model does (the original code '
                             'included it)')
    parser.add_argument('-frozen_resnet', action='store_true',
                        help='keep the pretrained ResNet in evaluation mode during training, so that its '
                             'BatchNorm layers keep the ImageNet statistics and the whole ResNet stays as '
                             'pretrained (the original code updated these statistics)')
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help='label smoothing of the loss, e.g. 0.1: the paper cites the regularization of '
                             'Vaswani et al., which includes it, but the code of its runs did not apply it')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None, help='the directory for the checkpoints')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'last'], default='best',
                        help='all: models/model.<epoch>.chkpt after every epoch; best: models/model.best.chkpt '
                             'whenever the validation result (-select_by) is the best so far; last: '
                             'models/model.last.chkpt after every epoch (the Amazon models of the paper)')
    parser.add_argument('-patience', type=int, default=0,
                        help='stop when the validation result has not improved for this many epochs '
                             '(0: train for -epoch epochs)')
    parser.add_argument('-select_by', choices=['model_metric', 'chain_accuracy'], default='model_metric',
                        help='the validation result of -save_mode best and -patience. model_metric, as in the '
                             'paper\'s runs: the accuracy of the training targets (of the tokens for the '
                             'sequential model, the attributes for the multitask models, the chains for '
                             'multiclass), the loss for multilabel. chain_accuracy: the full-chain accuracy of '
                             'the predicted validation chains, the same for all models (Rakuten, Kaggle)')
    parser.add_argument('-decode_steps', type=int, default=8,
                        help='-select_by chain_accuracy, sequential model: as translate.py -decode_steps')
    parser.add_argument('-no_check_chain', action='store_true',
                        help='-select_by chain_accuracy, sequential model: as translate.py -no_check_chain')
    parser.add_argument('-pretrained', type=str, default="", help='a checkpoint to continue from')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-num_workers', type=int, default=4, help='processes that load the images')
    parser.add_argument('-seed', type=int, required=True,
                        help='the seed of all random numbers: Python\'s string hashes (PYTHONHASHSEED), random, '
                             'NumPy and PyTorch on the CPU and GPUs (the initial weights, dropout, the order of '
                             'the training data, the random image crops and flips, the image loading processes); '
                             'with -kaggle_dir also the split and the SentencePiece model')
    parser.add_argument('-deterministic', action='store_true',
                        help='deterministic algorithms (slower) instead of the fastest ones: cuDNN\'s, and in '
                             'PyTorch 1.8 and later those of all operations; always on with -kaggle_dir')
    parser.add_argument('-tf32', action='store_true',
                        help='allow TensorFloat-32 on Ampere and newer GPUs: faster, but less precise than the '
                             'float32 of the experiments')
    parser.add_argument('-amp', action='store_true',
                        help='mixed precision: the model computes in float16 where PyTorch allows it (autocast), '
                             'with loss scaling, in training and validation; the weights stay float32. Faster on '
                             'GPUs with tensor cores, but other numbers than float32 (PyTorch 1.10 or later)')

    kaggle = parser.add_argument_group(
        'the Kaggle data split with -seed (instead of -data; see kaggle_data.py)')
    kaggle.add_argument('-kaggle_dir', help='the Kaggle Fashion Product Images data: styles.csv and images/')
    kaggle.add_argument('-kaggle_data_dir',
                        help='where the split, the SentencePiece model, the vocabularies and data.pt are written '
                             '(default: data/ next to the -save_model directory)')
    kaggle.add_argument('-valid_size', type=float, default=5000,
                        help='products in the validation set; a fraction of all products if below 1')
    kaggle.add_argument('-test_size', type=float, default=5000, help='products in the test set; likewise')
    kaggle.add_argument('-sp_vocab_size', type=int, default=4000,
                        help='pieces of the SentencePiece model of the training titles')
    kaggle.add_argument('-distinct_unk', action='store_true',
                        help='the token vocabularies start the tokens at index 4 (build_vocab.py -distinct_unk)')
    kaggle.add_argument('-missing_values', default='', metavar='VALUES',
                        help='comma-separated attribute values that mark a missing attribute, e.g. NA,None: '
                             'they are left out of the losses and the validation results (preprocess.py '
                             '-missing_values)')
    kaggle.add_argument('-exclude_ids', default='', metavar='IDS',
                        help='comma-separated ids of products to leave out of the data, e.g. 28319, a test record '
                             'with wrong attributes')

    opt = parser.parse_args()
    if (opt.data is None) == (opt.kaggle_dir is None):
        parser.error('give either -data or -kaggle_dir')
    if not 0 <= opt.seed < 2 ** 32:
        parser.error('-seed must be between 0 and 2**32 - 1')
    compat.seed_python_hash(opt.seed)  # runs the script again if PYTHONHASHSEED is not the seed
    if opt.kaggle_dir is not None:
        if opt.kaggle_data_dir is None:
            if opt.save_model is None:
                parser.error('-kaggle_dir needs -kaggle_data_dir or -save_model')
            opt.kaggle_data_dir = os.path.join(os.path.dirname(os.path.abspath(opt.save_model)), 'data')
        opt.image_dir = opt.image_dir or os.path.join(opt.kaggle_dir, 'images')
        opt.deterministic = True
    if opt.train_crop is None:
        opt.train_crop = 'random_resized_crop' if opt.kaggle_dir else 'random_crop'
    if opt.valid_crop is None:
        opt.valid_crop = 'center_crop' if opt.kaggle_dir else opt.train_crop
    opt.cuda = not opt.no_cuda
    if opt.amp and not (opt.cuda and compat.amp_supported()):
        parser.error('-amp needs a GPU and PyTorch 1.10 or later')
    opt.d_word_vec = opt.d_model
    opt.proj1_bias = not opt.no_proj1_bias
    compat.set_deterministic(opt.deterministic)
    compat.seed_everything(opt.seed)
    compat.set_tf32(opt.tf32)

    #========= Loading Dataset =========#
    if opt.kaggle_dir:
        opt.data = kaggle_data.prepare(opt.kaggle_dir, opt.kaggle_data_dir, opt.model_type, opt.seed,
                                       opt.valid_size, opt.test_size, opt.sp_vocab_size, opt.distinct_unk,
                                       opt.missing_values, opt.exclude_ids)
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
    if uses_image(opt.modality):
        try:
            check_image_dir(opt.image_dir, [opt.train_crop, opt.valid_crop], opt.crop_size)
        except ValueError as e:
            parser.error(str(e))
    if opt.select_by == 'chain_accuracy' and 'chains' not in data['valid']:
        parser.error('-select_by chain_accuracy needs the validation chains in data.pt: run preprocess.py again')
    tgt_format = TGT_FORMATS[opt.model_type]
    tgt_dict = data['dict']['tgt']

    #========= Preparing DataLoader =========#
    # With -kaggle_dir, the random crops and flips are drawn from the seed, the epoch and the product
    # (SeededRandomTransform), the training data is shuffled with the seed, and the image loading processes
    # are seeded from it. For the data of the paper's experiments, these random numbers come from the
    # generators seeded with -seed, as in the original code: the same seed gives the same crops with the
    # same -num_workers.
    seed = opt.seed if opt.kaggle_dir else None

    def transform(crop):
        if not uses_image(opt.modality):
            return None
        if seed is not None and crop != 'center_crop':
            return SeededRandomTransform(crop, opt.crop_size)
        return image_transform(crop, opt.crop_size)

    train_transform = transform(opt.train_crop)
    valid_transform = train_transform if opt.valid_crop == opt.train_crop else transform(opt.valid_crop)

    def loader(split, batch_size, shuffle, img_transform):
        return DataLoader(
            img_transform, opt.image_dir, data['dict'].get('src'), tgt_dict,
            image_insts=data[split]['img'] if uses_image(opt.modality) else None,
            src_insts=data[split]['src'] if uses_text(opt.modality) else None,
            tgt_insts=data[split]['tgt'], tgt_format=tgt_format, batch_size=batch_size, shuffle=shuffle,
            cuda=opt.cuda, num_workers=opt.num_workers, seed=seed)

    training_data = loader('train', opt.batch_size, True, train_transform)
    validation_data = loader('valid', int(opt.batch_size / 3), False, valid_transform)

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
        ''' With PAD token zero weight; missing values (Constants.MISSING) are ignored. '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, reduction='sum', ignore_index=Constants.MISSING)

    if opt.model_type == 'multilabel':
        crit = nn.BCEWithLogitsLoss()
    elif tgt_format == 'attributes':
        crit = [get_criterion(size) for size in opt.tgt_vocab_size]
    else:
        crit = get_criterion(opt.tgt_vocab_size)

    if opt.cuda:
        model = model.cuda()
        crit = [c.cuda() for c in crit] if isinstance(crit, list) else crit.cuda()

    chain_accuracy = None
    if opt.select_by == 'chain_accuracy':
        device = torch.device('cuda' if opt.cuda else 'cpu')
        chain_accuracy = ChainAccuracy(model, data, valid_transform, opt, device)

    train(model, training_data, validation_data, crit, optimizer, opt, chain_accuracy, compat.grad_scaler(opt.amp))


if __name__ == '__main__':
    main()
