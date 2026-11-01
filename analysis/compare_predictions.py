''' Test products where the MultiClass, MultiTask and Sequential models predict three different values of an
attribute and the Sequential model is right (Kaggle: attribute 4, counting from 0, the base colour).

    python analysis/compare_predictions.py -gold DATA/test.lbl -mc WORK_MC/pred_test.txt \
        -mt WORK_MT/pred_test.txt -seq WORK_S/pred_test.txt -out_dir DIR

Writes DIR/diff_mc.txt, DIR/diff_mlmc.txt and DIR/diff_ed.txt with "<line number>\\t<prediction>" for these
products. Predictions without the attribute ("None") are skipped. The original script (extract_diff.py)
also required the three values to come out of a Python 2 set in their order; that condition is left out.
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import read_lines  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-gold', required=True)
    parser.add_argument('-mc', required=True, help='MultiClass predictions')
    parser.add_argument('-mt', required=True, help='MultiTask predictions')
    parser.add_argument('-seq', required=True, help='Sequential predictions')
    parser.add_argument('-attribute', type=int, default=4)
    parser.add_argument('-out_dir', required=True)
    opt = parser.parse_args()

    gold, mc, mt, seq = (read_lines(p) for p in (opt.gold, opt.mc, opt.mt, opt.seq))
    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)
    targets = [(open(os.path.join(opt.out_dir, name), 'wb'), preds) for name, preds in
               (('diff_mlmc.txt', mt), ('diff_ed.txt', seq), ('diff_mc.txt', mc))]
    i = opt.attribute
    count = 0
    for a in range(len(mt)):
        values = [p[a].strip().split() for p in (mc, mt, seq)]
        lo = gold[a].strip().split()
        if any(len(v) <= i for v in values):
            continue
        if len(set(v[i] for v in values)) == 3 and values[2][i] == lo[i]:
            for target, preds in targets:
                target.write((str(a + 1) + "\t" + preds[a]).encode("utf-8"))
            count += 1
    for target, _ in targets:
        target.close()
    print('{} products'.format(count))


if __name__ == '__main__':
    main()
