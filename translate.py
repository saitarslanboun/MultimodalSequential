''' Predicts the attribute chains of products with a trained model; writes one chain per line. '''

import argparse
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

import rnmtplus.Constants as Constants
from DataLoader import CROPS, DataLoader, image_transform
from preprocess import convert_instance_to_idx_seq, read_instances_from_file
from rnmtplus import compat
from rnmtplus.Models import MODEL_TYPES, uses_image, uses_text
from rnmtplus.Translator import Translator


def make_batches(model_type, src_insts, n_insts, batch_size):
    ''' The batches (lists of product indices) for prediction.

    The multiclass and multilabel models predicted in consecutive slices of -batch_size in the original
    code. The titles of a batch are padded, and the padding positions take part in the average pooling of the
    text encodings, so these batches are kept. The other models predicted one product at a time; here the
    products whose titles have the same length are batched together, which needs no padding. '''
    if model_type in ('multiclass', 'multilabel') or src_insts is None:
        return [list(range(i, min(i + batch_size, n_insts))) for i in range(0, n_insts, batch_size)]
    by_length = OrderedDict()
    for i, inst in enumerate(src_insts):
        by_length.setdefault(len(inst), []).append(i)
    batches = []
    for indices in by_length.values():
        batches.extend(indices[j:j + batch_size] for j in range(0, len(indices), batch_size))
    return batches


def main():
    '''Main Function'''
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, help='the checkpoint (models/model.<epoch>.chkpt)')
    parser.add_argument('-vocab', required=True, help='the data.pt of the experiment (preprocess.py)')
    parser.add_argument('-img', help='image file names, one line per product (image models)')
    parser.add_argument('-src', help='titles, one line per product (text models)')
    parser.add_argument('-image_dir', help='the image directory (image models)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-batch_size', type=int, default=30,
                        help='batch size; for multiclass and multilabel, the batches of the original code')
    parser.add_argument('-model_type', choices=MODEL_TYPES,
                        help='the model type, if the checkpoint does not record it (original checkpoints)')
    parser.add_argument('-crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('-test_crop', choices=sorted(CROPS), default='random_resized_crop',
                        help='the random crop of the images, as in the experiment')
    parser.add_argument('-decode_steps', type=int, default=8,
                        help='sequential: the number of decoded tokens, attributes plus <EOS> '
                             '(Rakuten 8, Kaggle 9, Amazon 10)')
    parser.add_argument('-no_check_chain', action='store_true',
                        help='sequential: output the tokens up to the first <EOS> (Amazon) instead of "None" '
                             'for a sequence that does not end with its only <EOS> (Rakuten, Kaggle)')
    parser.add_argument('-save_attention', metavar='DIR',
                        help='sequential: save the attention weights of the last decoding step to '
                             'DIR/<line>.{txt,img}_attention.pt (for analysis/visualize_heatmap.py)')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-num_workers', type=int, default=4, help='processes that load the images')
    parser.add_argument('-deterministic', action='store_true',
                        help='deterministic cuDNN algorithms (slower) instead of the fastest ones')
    parser.add_argument('-tf32', action='store_true',
                        help='allow TensorFloat-32 on Ampere and newer GPUs: faster, but less precise than the '
                             'float32 of the experiments')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.check_chain = not opt.no_check_chain
    torch.backends.cudnn.benchmark = not opt.deterministic
    torch.backends.cudnn.deterministic = opt.deterministic
    compat.set_tf32(opt.tf32)

    translator = Translator(opt)
    model_type = translator.model_opt.model_type
    modality = translator.model_opt.modality
    if uses_image(modality) and not (opt.img and opt.image_dir):
        parser.error('the image models need -img and -image_dir')
    if uses_text(modality) and not opt.src:
        parser.error('the text models need -src')

    # Prepare DataLoader
    preprocess_data = compat.torch_load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    tgt_dict = preprocess_data['dict']['tgt']
    test_img_insts = test_src_insts = None
    if uses_image(modality):
        test_img_insts = read_instances_from_file(opt.img, preprocess_settings.max_word_seq_len)
    if uses_text(modality):
        test_src_word_insts = read_instances_from_file(opt.src, preprocess_settings.max_word_seq_len)
        test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, preprocess_data['dict']['src'])
    n_insts = len(test_src_insts if test_src_insts is not None else test_img_insts)

    test_data = DataLoader(
        image_transform(opt.test_crop, opt.crop_size) if uses_image(modality) else None, opt.image_dir,
        preprocess_data['dict'].get('src'), tgt_dict, image_insts=test_img_insts, src_insts=test_src_insts,
        cuda=opt.cuda, shuffle=False, num_workers=opt.num_workers,
        batches=make_batches(model_type, test_src_insts, n_insts, opt.batch_size))

    if model_type in ('sequential', 'multiclass'):
        inv_map = {v: k for k, v in tgt_dict.items()}
    else:
        inv_maps = [{v: k for k, v in tgt_dict[str(a)].items()} for a in range(len(tgt_dict.keys()))]
        vocab_sizes = [len(tgt_dict[str(a)]) for a in range(len(tgt_dict.keys()))]

    def chain(prediction):
        if model_type == 'sequential':
            if prediction is None:
                return "None"
            if opt.check_chain:
                seq = prediction[1:-1]
            elif Constants.EOS in prediction:
                seq = prediction[1:prediction.index(Constants.EOS)]
            else:
                seq = prediction[1:]
            return " ".join(inv_map[val] for val in seq)
        if model_type == 'multiclass':
            return inv_map[prediction]
        return " ".join(inv_maps[a][val] for a, val in enumerate(prediction))

    if opt.save_attention and not os.path.isdir(opt.save_attention):
        os.makedirs(opt.save_attention)
    lines = [None] * n_insts
    for indices, image, src, _ in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
        if model_type == 'sequential':
            predictions, attentions = translator.translate_batch(image, src)
            if opt.save_attention:
                for k, i in enumerate(indices):
                    for name, attention in zip(('txt', 'img'), attentions):
                        if attention is not None:
                            torch.save(attention[k:k + 1].cpu(), os.path.join(
                                opt.save_attention, '{}.{}_attention.pt'.format(i + 1, name)))
        else:
            predictions = translator.translate_batch(image, src, vocab_sizes if model_type == 'multilabel' else None)
        for i, prediction in zip(indices, predictions):
            lines[i] = chain(prediction)

    with open(opt.output, "wb") as target:
        for line in lines:
            target.write((line + "\n").encode("utf-8"))
    print('[Info] Finished.')


if __name__ == "__main__":
    main()
