
import gzip
import os
import sys
import urllib
import matplotlib as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcol
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from os import listdir, path
from parse import parse
from skimage.restoration import denoise_nl_means
from shutil import copyfile
from math import floor, isnan, ceil

import datetime

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf


NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
TEST_SIZE = 94
SEED = 1
TRAINING_BATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 512
NUM_EPOCHS = 100
RECORDING_STEP = 100


# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMG_FRAME_SIZE = 16

if len(sys.argv) != 2:
    print("usage: $0 logdir")
    sys.exit()

tf.app.flags.DEFINE_string('train_dir', sys.argv[1],
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

# Extract patches from a given image
def img_crop(im, w, h, frame, oversize):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]

    img_ul_corner = (0, 0)
    offset = 0

    if oversize:
        offset = frame
        imgwidth = int(floor(imgwidth / 3))
        imgheight = int(floor(imgheight / 3))
        img_ul_corner = (imgheight, imgwidth)

    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[(img_ul_corner[0]-offset+j):(img_ul_corner[0]+j+w+offset), (img_ul_corner[1]-offset+i):(img_ul_corner[1]+i+h+offset)]
            else:
                im_patch = im[(img_ul_corner[0]-offset+j):(img_ul_corner[0]+j+w+offset), (img_ul_corner[1]-offset+i):(img_ul_corner[1]+i+h+offset), :]

            list_patches.append(im_patch)
    return list_patches


def mirror_and_concat_img(img):

    ul = numpy.fliplr(numpy.flipud(img))
    ur = numpy.fliplr(numpy.flipud(img))
    u = numpy.flipud(img)
    l = numpy.fliplr(img)
    r = numpy.fliplr(img)
    b = numpy.flipud(img)
    br = numpy.fliplr(numpy.flipud(img))
    bl = numpy.fliplr(numpy.flipud(img))

    top_row = numpy.concatenate((ul, u, ur), axis=1)
    middle_row = numpy.concatenate((l, img, r), axis=1)
    bottom_row = numpy.concatenate((bl, b, br), axis=1)

    out = numpy.concatenate((top_row, middle_row, bottom_row), axis=0)

    return out


def extract_data(filename, num_images):
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(mirror_and_concat_img(imgs[i]), IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_FRAME_SIZE, True) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, 0, False) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def augment_training_data(data, labels):

    ia.seed(SEED)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode="edge",
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode="symmetric"
            )),

            iaa.SomeOf((0, 2),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

                           iaa.Sharpen(alpha=(0.9, 1.0), lightness=(0.75, 1.25)),
                           iaa.Emboss(alpha=(0.9, 1.0), strength=(0, 0.25)),

                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                           iaa.Dropout((0.01, 0.05), per_channel=0.5),

                           iaa.Add((-10, 10), per_channel=0.5),

                           iaa.AddToHueAndSaturation((-5, 5)),

                           iaa.ContrastNormalization((0.8, 1.1), per_channel=0.5),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),

                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    data8 = numpy.uint8(data*255.0)

    r0 = numpy.rot90(data8, 1, axes=(1, 2))
    r1 = numpy.rot90(data8, 2, axes=(1, 2))
    r2 = numpy.rot90(data8, 3, axes=(1, 2))

    m0 = numpy.flip(data8, 2)

    m0r0 = numpy.rot90(m0, 1, axes=(1, 2))
    m0r1 = numpy.rot90(m0, 2, axes=(1, 2))
    m0r2 = numpy.rot90(m0, 3, axes=(1, 2))

    m1 = numpy.flip(data8, 1)

    m1r0 = numpy.rot90(m1, 1, axes=(1, 2))
    m1r1 = numpy.rot90(m1, 2, axes=(1, 2))
    m1r2 = numpy.rot90(m1, 3, axes=(1, 2))

    rotated_and_mirrored = numpy.concatenate((data8, r0, r1, r2, m0, m0r0, m0r1, m0r2, m1, m1r0, m1r1, m1r2), axis=0)
    augmented_data8 = rotated_and_mirrored
    augmented_labels = numpy.tile(labels, (12,1))


    #for s in range(3):
    #    augmented_set = seq.augment_images(rotated_and_mirrored)
    #    augmented_data8 = numpy.concatenate((augmented_data8, augmented_set), axis=0)
    #    augmented_labels = numpy.concatenate((augmented_labels, labels), axis=0)

    augmented_data = numpy.float32(augmented_data8) / 255.0

    return augmented_data, augmented_labels


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            l = labels[idx]
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def make_img(predicted_img):
    img8 = img_float_to_uint8(predicted_img)
    image = Image.fromarray(img8, 'L').convert("RGBA")
    return image

def save_images(imgs, output_dir):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    cntr = 0
    for img in imgs:
        im = Image.fromarray(numpy.uint8(img * 255.0))
        im.save(output_dir + str("/") + str(cntr) + ".png")
        cntr += 1


# Get prediction for given input image
def get_prediction(classifier, img):

    data = numpy.asarray(img_crop(mirror_and_concat_img(img), IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_FRAME_SIZE, True))

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=data,
        num_epochs=1,
        batch_size=PREDICTION_BATCH_SIZE,
        shuffle=False)

    output_prediction = list(classifier.predict(input_fn=eval_input_fn))
    predicted_classes = [p["classes"] for p in output_prediction]

    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, predicted_classes)

    return img_prediction

# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(classifier, img):

    img_prediction = get_prediction(classifier, img)
    cimg = concatenate_images(img, img_prediction)

    return cimg

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(classifier, img):

    img_prediction = get_prediction(classifier, img)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

def classify_files_in_dir(classifier, input_dir, output_dir, count):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    file_names = [f for f in sorted(listdir(input_dir)) if path.isfile(path.join(input_dir, f))]
    for i in range(count):
        filename = file_names[i]

        img = mpimg.imread(input_dir + filename)

        pimg = get_prediction_with_groundtruth(classifier, img)
        Image.fromarray(pimg).save(output_dir + "prediction_" + filename)
        oimg = get_prediction_with_overlay(classifier, img)
        oimg.save(output_dir + "overlay_" + str(filename))
        pimg = get_prediction(classifier, img)
        pimg = make_img(pimg)
        pimg.save(output_dir + "mask_" + str(filename))

def main(argv=None):

    timestamp = "{}".format(datetime.datetime.now().strftime("%d-%B-%H:%M:%S"))
    data_dir = 'training/'
    test_dir = 'test_images/'
    train_data_path = data_dir + 'images/'
    train_labels_path = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_path, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_path, TRAINING_SIZE)
    train_data, train_labels = augment_training_data(train_data, train_labels)

    print("Patch # after augmentation: ", train_data.shape[0])

    #save_images(train_data, "augmented_training_data")
    #sys.exit()

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    #print('Balancing training data...')
    #min_c = min(c0, c1)
    #idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    #idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    #new_indices = idx0[0:min_c] + idx1[0:min_c]

    #train_data = train_data[new_indices,:,:,:]
    #train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(features, labels, mode):

        conv1 = tf.layers.conv2d(
            inputs=features,
            filters=32,
            kernel_size=[3, 3],
            padding="SAME",
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2], padding='SAME')

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="SAME",
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2], padding='SAME')

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="SAME",
            activation=tf.nn.relu
        )

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2], padding='SAME')

        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=256,
            kernel_size=[3, 3],
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=SEED),
            bias_initializer=tf.constant_initializer(0.1)
        )

        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[2, 2], padding='SAME')

        conv5 = tf.layers.conv2d(
            inputs=pool4,
            filters=256,
            kernel_size=[3, 3],
            padding="SAME",
            activation=tf.nn.relu
        )

        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=[2, 2], padding='SAME')

        #print('data ' + str(tf.shape(features)))
        #print('conv1 ' + str(tf.shape(conv1)))
        #print('pool1 ' + str(tf.shape(pool1)))
        #print('conv2 ' + str(tf.shape(conv2)))
        #print('pool2 ' + str(tf.shape(pool2)))
        #print('conv3 ' + str(tf.shape(conv3)))
        #print('pool3 ' + str(tf.shape(pool3)))
        #print('conv4 ' + str(tf.shape(conv4)))
        #print('pool4 ' + str(tf.shape(pool4)))
        #print('conv5 ' + str(tf.shape(conv5)))
        #print('pool5 ' + str(tf.shape(pool5)))

        flattened = tf.layers.flatten(pool5)

        fc1 = tf.layers.dense(
            inputs=flattened,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=SEED),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))

        dropout = tf.layers.dropout(inputs=fc1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, seed=SEED)

        logits = tf.layers.dense(
            inputs=dropout,
            units=2,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=SEED),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.softmax_cross_entropy(labels, logits)
        regularization_loss = tf.losses.get_regularization_loss()
        loss += regularization_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                0.01,
                tf.train.get_global_step() * TRAINING_BATCH_SIZE,
                train_size,
                0.98,
                staircase=True)

            momentum = tf.train.exponential_decay(
                0.005,
                tf.train.get_global_step() * TRAINING_BATCH_SIZE,
                train_size,
                1.00,
                staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('momentum', momentum)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
                #tf.summary.image(var.name, var)

            summary_hook = tf.train.SummarySaverHook(RECORDING_STEP, summary_op=tf.summary.merge_all())
            logging_hook = tf.train.LoggingTensorHook({"epoch": tf.train.get_global_step() / tf.constant(ceil(train_size / TRAINING_BATCH_SIZE), dtype=tf.int64), "learning_rate": learning_rate, "momentum": momentum }, every_n_iter=RECORDING_STEP)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,  training_hooks=[summary_hook, logging_hook])

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    classifier = tf.estimator.Estimator(model_fn=model, model_dir=FLAGS.train_dir + "/" + timestamp)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=TRAINING_BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        shuffle=True
    )

    classifier.train(
        input_fn=train_input_fn,
        steps=None
    )

    print ("Running prediction on training set")
    classify_files_in_dir(classifier, train_data_path, timestamp + "_predictions_training/", TRAINING_SIZE)

    print ("Running prediction on test set")
    classify_files_in_dir(classifier, test_dir, timestamp + "_predictions_test/", TEST_SIZE)

    copyfile(sys.argv[0], timestamp + "_predictions_test/model.py")


if __name__ == '__main__':
    tf.app.run()
