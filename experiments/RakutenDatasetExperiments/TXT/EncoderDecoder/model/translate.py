''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm
from rnmtplus.Translator import Translator
from DataLoader import DataLoader
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
from torchvision import transforms

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_data = DataLoader(
        preprocess_data['dict']['src'],
        preprocess_data['dict']['tgt'],
        src_insts=test_src_insts,
        cuda=opt.cuda,
        shuffle=False,
        batch_size=1)

    translator = Translator(opt)
    translator.model.eval()

    inv_map = {v: k for k, v in preprocess_data['dict']['tgt'].items()}

    target = open(opt.output, "wb")
    for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
        seq = translator.translate_batch(batch)
        if seq is None:
            line = "None\n"
            target.write(line.encode("utf-8"))
            continue
        seq = seq[1:-1]
        line = [inv_map[val] for val in seq]
        line = " ".join(line) + "\n"
        target.write(line.encode("utf-8"))
    target.close()
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
