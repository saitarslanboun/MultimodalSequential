''' Converts the data files of an experiment into data.pt, the input of train.py and translate.py.

The targets depend on the model:
- sequential: the attribute chain as a token sequence <BOS> value ... value <EOS>,
- multiclass: the index of the whole chain in the class vocabulary,
- multilabel and the multitask models: one index per attribute, from one vocabulary per attribute.
The vocabularies are built with build_vocab.py.
'''

import argparse
import io
import json
import sys

import torch

from rnmtplus.Models import MODEL_TYPES

TGT_FORMATS = {'sequential': 'sequence', 'multiclass': 'chain', 'multilabel': 'attributes',
               'multitask': 'attributes', 'multitask_att': 'attributes', 'multitask_slfatt': 'attributes'}


def read_lines(inst_file):
    ''' The lines of a UTF-8 file, as codecs.open(inst_file, encoding="utf-8").readlines() returns them. '''
    with io.open(inst_file, encoding='utf-8', newline='') as f:
        return f.read().splitlines(True)


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence; unknown words become <UNK>. '''
    idx_seqs = []
    for inst in word_insts:
        idx_seq = []
        for token in inst:
            if token in word2idx:
                idx_seq.append(word2idx[token])
            else:
                idx_seq.append(word2idx['<UNK>'])
        idx_seqs.append(idx_seq)
    return idx_seqs


def convert_target_instance_to_idx(line_insts, line2idx):
    ''' Mapping lines (whole attribute chains) to their class index; unknown chains become <UNK>. '''
    idx = []
    for inst in line_insts:
        chain = inst.strip()
        idx.append(line2idx[chain] if chain in line2idx else line2idx['<UNK>'])
    return idx


def convert_target_instance_to_attribute_idx(line_insts, line2idx):
    ''' Mapping lines to one index per attribute, from the vocabulary of each attribute (line2idx['0'],
    line2idx['1'], ...); unknown values become <UNK>. Returns one list per attribute. '''
    idx = []
    for inst in line_insts:
        tokens = inst.strip().split()
        if len(idx) == 0:
            for a in range(len(tokens)):
                idx.append([])
        for a in range(len(tokens)):
            vocab = line2idx[str(a)]
            idx[a].append(vocab[tokens[a]] if tokens[a] in vocab else vocab['<UNK>'])
    return idx


def read_image_instances_from_file(inst_file):
    ''' Convert file into lists '''
    return [sent.strip() for sent in read_lines(inst_file)]


def read_target_instances_from_file(inst_file):
    ''' Convert target file into class lists '''
    with io.open(inst_file, encoding='utf-8') as f:
        line_insts = list(f)
    print('[Info] Get {} instances from {}'.format(len(line_insts), inst_file))
    return line_insts


def read_instances_from_file(inst_file, max_sent_len):
    ''' Convert file into word seq lists: <BOS> words <EOS>, or None for an empty line. '''
    word_insts = []
    trimmed_sent_count = 0
    with io.open(inst_file, encoding='utf-8') as f:
        for sent in f:
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]
            if word_inst:
                word_insts += [['<BOS>'] + word_inst + ['<EOS>']]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    return word_insts


def load_json(path):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='preprocess.py')
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES)
    parser.add_argument('-train_img', help='image file names (image models)')
    parser.add_argument('-train_src', help='titles (text models)')
    parser.add_argument('-train_tgt', required=True, help='attribute chains')
    parser.add_argument('-valid_img')
    parser.add_argument('-valid_src')
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-src_vocab', help='title vocabulary (text models)')
    parser.add_argument('-tgt_vocab', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=1000)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2
    tgt_format = TGT_FORMATS[opt.model_type]
    image = opt.train_img is not None
    text = opt.train_src is not None
    if not (image or text):
        parser.error('give -train_img/-valid_img, -train_src/-valid_src/-src_vocab, or both')
    if image != (opt.valid_img is not None) or text != (opt.valid_src is not None) \
            or text != (opt.src_vocab is not None):
        parser.error('-train_img needs -valid_img; -train_src needs -valid_src and -src_vocab')

    def read_split(img, src, tgt, name):
        img_insts = read_image_instances_from_file(img) if image else None
        src_word_insts = read_instances_from_file(src, opt.max_word_seq_len) if text else None
        if tgt_format == 'sequence':
            tgt_insts = read_instances_from_file(tgt, opt.max_word_seq_len)
        else:
            tgt_insts = read_target_instances_from_file(tgt)
        counts = [len(insts) for insts in (img_insts, src_word_insts, tgt_insts) if insts is not None]
        if len(set(counts)) != 1:
            print('[Warning] The {} instance count is not equal!'.format(name))
            sys.exit(1)
        return img_insts, src_word_insts, tgt_insts

    train_img_insts, train_src_word_insts, train_tgt_insts = read_split(
        opt.train_img, opt.train_src, opt.train_tgt, 'training')
    valid_img_insts, valid_src_word_insts, valid_tgt_insts = read_split(
        opt.valid_img, opt.valid_src, opt.valid_tgt, 'validation')

    # Vocabularies (build_vocab.py)
    src_word2idx = load_json(opt.src_vocab) if text else None
    tgt_word2idx = load_json(opt.tgt_vocab)

    if text:
        print('[Info] Convert source word instances into sequences of word index.')
        train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
        valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)
    if tgt_format == 'sequence':
        print('[Info] Convert target word instances into sequences of word index.')
        convert = convert_instance_to_idx_seq
    else:
        print('[Info] Convert target line instances into line index.')
        convert = convert_target_instance_to_idx if tgt_format == 'chain' else \
            convert_target_instance_to_attribute_idx
    train_tgt_insts = convert(train_tgt_insts, tgt_word2idx)
    valid_tgt_insts = convert(valid_tgt_insts, tgt_word2idx)

    data = {'settings': opt, 'dict': {'tgt': tgt_word2idx}, 'train': {'tgt': train_tgt_insts},
            'valid': {'tgt': valid_tgt_insts}}
    if image:
        data['train']['img'] = train_img_insts
        data['valid']['img'] = valid_img_insts
    if text:
        data['dict']['src'] = src_word2idx
        data['train']['src'] = train_src_insts
        data['valid']['src'] = valid_src_insts

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')


if __name__ == '__main__':
    main()
