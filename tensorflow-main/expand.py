import util
import numpy as np
from PIL import Image

train_data = util.load_train_img(tiling=False)
train_labels = util.load_train_lbl(tiling=False)

train_labels = np.around(train_labels)
train_labels = train_labels.astype('int32')

# EXPAND to 608 x 608
train_data = np.pad(train_data, ((0, 0), (104, 104), (104, 104), (0, 0)), 'reflect')
train_labels = np.pad(train_labels, ((0, 0), (104, 104), (104, 104)), 'reflect')

for i in range(train_data.shape[0]):
    img = train_data[i]
    mask = train_labels[i]

    img = util.img_float_to_uint8(img)
    mask = util.img_float_to_uint8(mask)

    Image.fromarray(img).save('expand_data/training/images/satImage_{:03}.png'.format(i+1))
    Image.fromarray(mask).save('expand_data/training/groundtruth/satImage_{:03}.png'.format(i+1))