''' Test products that the text-only model predicts correctly and the text+image model does not (Section 5.2).

    python analysis/find_errors.py -txt WORK_TXT/pred_test.txt -mul WORK_TXT_IMG/pred_test.txt \
        -gold DATA/test.lbl -titles DATA/test.enc.tit -images DATA/test.img -out_dir DIR

Writes for the first -max (50) such products, one line each: DIR/ind (the line number of the product),
DIR/txt and DIR/mul (the two predictions), DIR/title and DIR/image.
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import read_lines  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-txt', required=True, help='predictions of the text-only model')
    parser.add_argument('-mul', required=True, help='predictions of the text+image model')
    parser.add_argument('-gold', required=True)
    parser.add_argument('-titles', required=True)
    parser.add_argument('-images', required=True)
    parser.add_argument('-out_dir', required=True)
    parser.add_argument('-max', type=int, default=50)
    opt = parser.parse_args()

    txt, mul, lbl = read_lines(opt.txt), read_lines(opt.mul), read_lines(opt.gold)
    titles, images = read_lines(opt.titles), read_lines(opt.images)
    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)
    names = ('ind', 'txt', 'mul', 'title', 'image')
    targets = dict((name, open(os.path.join(opt.out_dir, name), 'wb')) for name in names)
    count = 0
    for a in range(len(txt)):
        if txt[a] == lbl[a] and mul[a] != lbl[a]:
            lines = {'ind': str(a + 1) + '\n', 'txt': txt[a], 'mul': mul[a], 'title': titles[a], 'image': images[a]}
            for name in names:
                targets[name].write(lines[name].encode('utf-8'))
            count += 1
        if count == opt.max:
            break
    for target in targets.values():
        target.close()
    print('{} products'.format(count))


if __name__ == '__main__':
    main()
