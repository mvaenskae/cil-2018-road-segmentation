#!/usr/bin/env python

from jimmy import *
import numpy as np
from PIL import Image

output_dir = 'augmentations'

imgs,gts = read_images_plus_labels()

aug_img = np.empty(imgs.shape)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

cntr = 0
for i,g in zip(imgs, gts):
    img,_ = epoch_augmentation(i,g,0)
    aug_img[cntr,:,:,:] = img

    im = Image.fromarray(np.uint8(img * 255.0))
    im.save(output_dir + str("/") + str(cntr) + ".png")
    cntr += 1

#save_images(aug_img,'augmented')
