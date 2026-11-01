''' Heat maps of the image attention of the Sequential model.

    python translate.py ... -save_attention ATT
    python analysis/visualize_heatmap.py -attention ATT/<line>.img_attention.pt -image IMAGES/<name>.jpg

Writes <out_prefix><head>.png (default out_0.png, out_1.png, ...) for every attention head: the image
(resized to 224x224) with the attention of the head, summed over the decoder positions and upsampled from the
14x14 grid. Needs OpenCV (cv2) and scikit-image.
'''

import argparse

import cv2
import numpy
import skimage.transform
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-attention', required=True, help='an img_attention.pt of translate.py')
    parser.add_argument('-image', required=True)
    parser.add_argument('-out_prefix', default='out_')
    opt = parser.parse_args()

    mat = torch.load(opt.attention, map_location='cpu')  # 1 x heads x decoder positions x 196
    img = cv2.resize(cv2.imread(opt.image), (224, 224))
    for a in range(mat.shape[1]):
        current_alpha = mat[:, a, :, :]
        ncalpha = torch.zeros(current_alpha[:, 0, :].shape)
        for b in range(current_alpha.shape[1]):
            ncalpha += current_alpha[:, b, :]
        ncalpha = ncalpha.view(14, 14)
        alpha = skimage.transform.pyramid_expand(ncalpha.detach().cpu().numpy(), upscale=16, sigma=8)
        alpha = numpy.uint8(alpha * 255)
        alpha = cv2.applyColorMap(alpha, cv2.COLORMAP_JET)
        alpha = numpy.uint8(alpha + img)
        cv2.imwrite(opt.out_prefix + str(a) + ".png", alpha)


if __name__ == '__main__':
    main()
