import matplotlib.image as mpimg
import os
import numpy
import errno
import constants

'''
Utility large patches
'''


def crete_patches_large(data, patch_size, stride, padding, is_mask=False):
    num_imgs = data.shape[0]
    img_patches = []

    for i in range(num_imgs):
        img_patches.append(_crop_patches_large(data[i], patch_size, patch_size, stride, padding, is_mask))

    img_patches = numpy.asarray(img_patches)
    if is_mask:
        data = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
        img_patches = numpy.asarray([_value_to_class(numpy.mean(data[i])) for i in range(len(data))])
        img_patches = img_patches.astype(numpy.float32)
    else:
        img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches


def _crop_patches_large(img, w, h, stride, padding, is_mask, mode='reflect'):
    img_patches = []
    img_width = img.shape[0]
    img_height = img.shape[1]

    if is_mask:
        # mask
        img = numpy.lib.pad(img, ((padding, padding), (padding, padding)), mode)
        for i in range(padding, img_height + padding, stride):
            for j in range(padding, img_width + padding, stride):
                im_patch = img[j - padding:j + w + padding, i - padding:i + h + padding]
                img_patches.append(im_patch)
    else:
        # img
        img = numpy.lib.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode)
        for i in range(padding, img_height + padding, stride):
            for j in range(padding, img_width + padding, stride):
                im_patch = img[j - padding:j + w + padding, i - padding:i + h + padding, :]
                img_patches.append(im_patch)

    return img_patches


'''
Pure utility funcitons
'''


def get_file_names():
    data_dir = 'data/test_images/'
    names = [filename for filename in os.listdir(data_dir)]
    return names


def read_img(path):
    img = mpimg.imread(path)
    data = numpy.asarray(_img_crop(img, constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE))
    return data


def create_prediction_dir(prediction_test_dir):
    print("Running prediction")
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * constants.PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def label_to_img_full(w, h, labels):
    array_labels = numpy.zeros([w, h])
    for i in range(h):
        for j in range(w):
            if labels[i][j][0] > 0.5:
                l = 0
            else:
                l = 1
            array_labels[i, j] = l
    return array_labels


def label_to_img_inverse(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:
                l = 0
            else:
                l = 1
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def one_hot_to_num(lbl):
    return numpy.argmax(lbl, axis=1)


def channel_first(tensor):
    return numpy.rollaxis(tensor, 3, 1)


'''
TRAIN DATA
'''


def load_train_lbl(tiling=True):
    data_dir = 'data/training/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_labels = _extract_labels(train_labels_filename, constants.N_IMAGES, tiling)
    return train_labels


def load_train_img(tiling=True):
    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'

    # Extract it into numpy arrays.
    train_data = _extract_data(train_data_filename, constants.N_IMAGES, tiling)
    return train_data


'''
TEST DATA
'''


def load_test_data(tiling=True):
    data_dir = 'data/test_images/'
    imgs = []
    for filename in os.listdir(data_dir):
        path = data_dir + filename
        if os.path.isfile(path):
            # print('Loading ' + path)
            img = mpimg.imread(path)
            imgs.append(img)
        else:
            print('File ' + path + ' does not exist')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if tiling:
        imgs = _cut_tiles_img(imgs)

    return numpy.asarray(imgs)


'''
PRIVATE METHODS
'''


def _extract_data(filename, num_images, tiling=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            # print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_filename)

    if tiling:
        imgs = _cut_tiles_img(imgs)

    return numpy.asarray(imgs)


# Extract label images
def _extract_labels(filename, num_images, tiling=True):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            # print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_filename)

    if tiling:
        data = _cut_tiles_lbl(gt_imgs)
        labels = numpy.asarray([_value_to_class(numpy.mean(data[i])) for i in range(len(data))])
    else:
        labels = numpy.asarray(gt_imgs)

    return labels.astype(numpy.float32)


# Extract patches from a given image
def _img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


# Assign a label to a patch v
def _value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def _cut_tiles_img(imgs):
    num_images = len(imgs)
    img_patches = [_img_crop(imgs[i], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return data


def _cut_tiles_lbl(gt_imgs):
    num_images = len(gt_imgs)
    gt_patches = [_img_crop(gt_imgs[i], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE) for i in
                  range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    return data
