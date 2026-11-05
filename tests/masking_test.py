''' Checks that the models leave out the image of a product without one (MaskedImages, train.py
-mask_missing_images), for every model type with image input (img, txt+img), with random weights:
- with all images present, the outputs are those of a plain image batch, bit for bit (the original code path);
- the image of an absent product does not change any output: the scores of the forward pass and the
  predictions are the same with another image in its place;
- the image of a present product does change its scores (a check of the check).
And the same for several views per product (train.py -max_views; here 3 view slots, with a random view
embedding), with a product that has no view:
- the image in the slot of an absent view does not change any output;
- a present view does change the scores of its product, and so does moving a view to another slot (the view
  embedding);
- with the view embedding at zero (as at the start of training), a product with one view in the first slot gets
  the scores of the model for one image (compared in float64: in float32, attending to the regions of 3 slots
  rounds differently from attending to those of one, which the random weights amplify).

    python tests/masking_test.py [-cuda]

Prints one line per model and check and exits with status 1 if any check fails.
'''

import argparse
import copy
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from rnmtplus.Models import MODEL_TYPES, MaskedImages, build_model  # noqa: E402
from rnmtplus.Translator import Translator  # noqa: E402

N_SRC, N_TGT, ATTRIBUTE_SIZES = 20, 30, [5, 7, 4]
N_VIEWS = 3


def model_options(model_type, modality, max_views=1):
    opt = argparse.Namespace(model_type=model_type, modality=modality, d_model=1024, d_word_vec=1024, n_layers=2,
                             n_head=8, dropout=0.1, src_vocab_size=N_SRC, mask_padding=True, proj1_bias=True,
                             unused_slf_attention=False, max_views=max_views)
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


def inputs(modality, device):
    ''' A title batch (or None) and target prefixes for 3 products. '''
    src = torch.randint(4, N_SRC, (3, 6), device=device) if 'txt' in modality else None
    if src is not None:
        src[2, 4:] = 0  # padding
    tgt = torch.randint(4, N_TGT, (3, 5), device=device)
    tgt[:, 0] = 1  # <BOS>
    return src, tgt


def view_problems(model_type, modality, device):
    ''' The failed checks of the views (see above). '''
    problems = []
    torch.manual_seed(0)
    model_opt = model_options(model_type, modality, N_VIEWS)
    model = build_model(model_opt, pretrained=False).to(device).eval()
    single_opt = model_options(model_type, modality)
    single = build_model(single_opt, pretrained=False).to(device).eval()
    state = model.state_dict()
    single.load_state_dict({k: v for k, v in state.items() if k != 'view_embedding'})
    images = torch.randn(3, N_VIEWS, 3, 224, 224, device=device)
    src, tgt = inputs(modality, device)
    first = torch.zeros(3, N_VIEWS, dtype=torch.uint8, device=device)
    first[:, 0] = 1
    model64, single64, images64 = copy.deepcopy(model).double(), copy.deepcopy(single).double(), images.double()
    if not torch.allclose(scores(model64, model_type, MaskedImages(images64, first), src, tgt),
                          scores(single64, model_type, images64[:, 0], src, tgt), rtol=1e-9, atol=1e-9):
        problems.append('one view with a zero view embedding differs from one image')
    del model64, single64, images64
    with torch.no_grad():
        model.view_embedding.normal_()
    translator = Translator.from_model(
        model, model_opt, argparse.Namespace(decode_steps=4, check_chain=False, amp=False), device)
    present = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.uint8, device=device)  # product 3: none
    other = images.clone()
    for b, v in [(0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]:  # other images in the slots of the absent views
        other[b, v] = torch.randn(3, 224, 224, device=device)
    masked = scores(model, model_type, MaskedImages(images, present), src, tgt)
    if not torch.equal(masked, scores(model, model_type, MaskedImages(other, present), src, tgt)):
        problems.append('an absent view changes the scores')
    if predictions(translator, model_type, MaskedImages(images, present), src) != \
            predictions(translator, model_type, MaskedImages(other, present), src):
        problems.append('an absent view changes the predictions')
    changed = images.clone()
    changed[0, 1] = torch.randn(3, 224, 224, device=device)
    if torch.equal(masked[0], scores(model, model_type, MaskedImages(changed, present), src, tgt)[0]):
        problems.append('a present view does not change the scores')
    moved, moved_present = images.clone(), present.clone()
    moved[1, 1], moved_present[1] = images[1, 0], torch.tensor([0, 1, 1], dtype=torch.uint8, device=device)
    if torch.equal(masked[1], scores(model, model_type, MaskedImages(moved, moved_present), src, tgt)[1]):
        problems.append('moving a view to another slot does not change the scores')
    return problems


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
            src, tgt = inputs(modality, device)
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
            print('%-8s %-17s %-6s %s' % (modality, model_type, 'image', status))
            problems = view_problems(model_type, modality, device)
            failures += bool(problems)
            print('%-8s %-17s %-6s %s' % (modality, model_type, 'views', 'ok' if not problems else
                                           'FAILED: ' + '; '.join(problems)))
            sys.stdout.flush()
    print('%d failed' % failures if failures else 'all %d passed' % (4 * len(MODEL_TYPES)))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
