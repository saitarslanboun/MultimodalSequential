from PIL import Image
from torchvision import transforms

import torch
import skimage
import matplotlib.pyplot as plt
import cv2
import numpy

mat = torch.load("img_attention.pt")
img = cv2.imread("324529.jpg")
img = cv2.resize(img, (224, 224))

for a in range(8):
	current_alpha = mat[:, a, :, :]
	ncalpha = torch.zeros(current_alpha[:, 0, :].shape).cuda()
	for b in range(8):
		ncalpha += current_alpha[:, b, :]
	ncalpha = ncalpha.view(14, 14)
	alpha = skimage.transform.pyramid_expand(ncalpha.detach().cpu().numpy(), upscale=16, sigma=8)
	alpha = numpy.uint8(alpha * 255)
	alpha = cv2.applyColorMap(alpha, cv2.COLORMAP_JET)
	alpha = numpy.uint8(alpha + img)
	name = "out_" + str(a) + ".png"
	cv2.imwrite(name, alpha)
