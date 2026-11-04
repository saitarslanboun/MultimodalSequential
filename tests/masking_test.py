''' Checks that the models leave out the image of a product without one (MaskedImages, train.py
-mask_missing_images), for every model type with image input (img, txt+img), with random weights:
- with all images present, the outputs are those of a plain image batch, bit for bit (the original code path);
- the image of an absent product does not change any output: the scores of the forward pass and the
  predictions are the same with another image in its place;
- the image of a present product does change its scores (a check of the check).

    python tests/masking_test.py [-cuda]

Prints one line per model and exits with status 1 if any check fails.
'''

import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from rnmtplus.Models import MODEL_TYPES, MaskedImages, build_model  # noqa: E402
from rnmtplus.Translator import Translator  # noqa: E402

N_SRC, N_TGT, ATTRIBUTE_SIZES = 20, 30, [5, 7, 4]


def model_options(model_type, modality):
    opt = argparse.Namespace(model_type=model_type, modality=modality, d_model=1024, d_word_vec=1024, n_layers=2,
                             n_head=8, dropout=0.1, src_vocab_size=N_SRC, mask_padding=True, proj1_bias=True,
                             unused_slf_attention=False)
    if model_type in ('sequential', 'multiclass'):
        opt.tgt_vocab_size = N_TGT
    elif model_type == 'multilabel':
        opt.tgt_vocab_size, opt.tgt_vocab_sizes = sum(ATTRIBUTE_SIZES), ATTRIBUTE_SIZES
    else:
        opt.tgt_vocab_size = ATTRIBUTE_SIZES
    return opt


def scores(model, model_type, image, src, tgt):
    ''' The scores of the forward pass, per product: batch x everything else. '''
    with torch.no_grad():
        if model_type == 'sequential':
            out = model(image, src, tgt)
            return out.view(tgt.size(0), -1)
        out = model(image, src)
        return torch.cat(out, dim=1) if isinstance(out, list) else out


def predictions(translator, model_type, image, src):
    out = translator.translate_batch(image, src, ATTRIBUTE_SIZES if model_type == 'multilabel' else None)
    return out[0] if model_type == 'sequential' else out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-cuda', action='store_true')
    opt = parser.parse_args()
    device = torch.device('cuda' if opt.cuda else 'cpu')
    failures = 0
    for modality in ('img', 'txt+img'):
        for model_type in MODEL_TYPES:
            torch.manual_seed(0)
            model_opt = model_options(model_type, modality)
            model = build_model(model_opt, pretrained=False).to(device).eval()
            translator = Translator.from_model(
                model, model_opt, argparse.Namespace(decode_steps=4, check_chain=False, amp=False), device)
            images = torch.randn(3, 3, 224, 224, device=device)
            other = images.clone()
            other[1] = torch.randn(3, 224, 224, device=device)  # another image for the second product
            src = torch.randint(4, N_SRC, (3, 6), device=device) if 'txt' in modality else None
            if src is not None:
                src[2, 4:] = 0  # padding
            tgt = torch.randint(4, N_TGT, (3, 5), device=device)
            tgt[:, 0] = 1  # <BOS>
            all_present = torch.ones(3, dtype=torch.uint8, device=device)
            second_absent = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)
            problems = []

            plain = scores(model, model_type, images, src, tgt)
            if not torch.equal(plain, scores(model, model_type, MaskedImages(images, all_present), src, tgt)):
                problems.append('all present differs from a plain batch')
            masked = scores(model, model_type, MaskedImages(images, second_absent), src, tgt)
            if not torch.equal(masked, scores(model, model_type, MaskedImages(other, second_absent), src, tgt)):
                problems.append('the absent image changes the scores')
            if predictions(translator, model_type, MaskedImages(images, second_absent), src) != \
                    predictions(translator, model_type, MaskedImages(other, second_absent), src):
                problems.append('the absent image changes the predictions')
            if torch.equal(plain[1], scores(model, model_type, other, src, tgt)[1]):
                problems.append('a present image does not change the scores')
            if torch.equal(masked[1], plain[1]):
                problems.append('masking the image does not change the scores')
            status = 'ok' if not problems else 'FAILED: ' + '; '.join(problems)
            failures += bool(problems)
            print('%-8s %-17s %s' % (modality, model_type, status))
            sys.stdout.flush()
    print('%d failed' % failures if failures else 'all %d passed' % (2 * len(MODEL_TYPES)))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
