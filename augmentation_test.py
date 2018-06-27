#!/usr/bin/env python

from jimmy import *
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image

output_dir = 'augmentations'

imgs,gts = read_images_plus_labels()

aug_img = np.empty(imgs.shape)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

cntr = 0
for i,g in zip(imgs, gts):
    print("augmenting...")
    #sigma = np.mean(estimate_sigma(i, multichannel=True))
    img,_ = epoch_augmentation(i,g,0)

    #patch_kw = dict(patch_size=3,
    #                patch_distance=3,
    #                multichannel=True)

    #img = denoise_nl_means(i, h=0.85*sigma, fast_mode=False, **patch_kw)

    aug_img[cntr,:,:,:] = img

    im = Image.fromarray(np.uint8(img * 255.0))
    im.save(output_dir + str("/") + str(cntr) + ".png")
    cntr += 1
    

