from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import constants
import tensorflow as tf
import util
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

NUM_CLASSES = 2
IMG_SIZE = 608


def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN.
    :param features: input X fed to the estimator
    :param labels: input Y fed to the estimator
    :param mode: TRAIN, EVAL, PREDICT
    :return: tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    """

    # Input Layer
    # 4-D tensor: [batch_size, width, height, channels]
    input_layer = features["x"]

    # Image augmentation
    # WARNING: Have to flip the maps as well on the full images
    if mode == tf.estimator.ModeKeys.TRAIN:
        # FLIP UP DOWN
        flip_ud = lambda x: tf.image.random_flip_up_down(x)
        input_layer = tf.map_fn(flip_ud, input_layer)
        labels = tf.map_fn(flip_ud, labels)

        # FLIP LEFT RIGHT
        flip_lr = lambda x: tf.image.flip_left_right(x)
        input_layer = tf.map_fn(flip_lr, input_layer)
        labels = tf.map_fn(flip_lr, labels)

        # BRIGHTNESS
        # bright = lambda x: tf.image.random_brightness(x, max_delta=0.00005)
        # input_layer = tf.map_fn(bright, input_layer)

        # CONTRAST
        contrast = lambda x: tf.image.random_contrast(x, lower=0.7, upper=1.1)
        input_layer = tf.map_fn(contrast, input_layer)

        # HUE
        # hue = lambda x: tf.image.random_hue(x, max_delta=0.1)
        # input_layer = tf.map_fn(hue, input_layer)

        # # SATURATION
        # satur = lambda x: tf.image.random_saturation(x, lower=0.1, upper=0.15)
        # input_layer = tf.map_fn(satur, input_layer)

        tf.summary.image('Augmentation', input_layer, max_outputs=4)
        # remove the meaningless channel axis after augmentation on maps
        labels = tf.squeeze(labels, axis=-1)

    # Channel first now
    input_layer = tf.transpose(input_layer, [0, 3, 1, 2])

    def conv(x, filters, kernel, name):
        return tf.layers.conv2d(inputs=x,
                                filters=filters,
                                kernel_size=[kernel, kernel],
                                padding="same",
                                activation=tf.nn.relu,
                                name=name,
                                data_format='channels_first')

    def pool(x, pool, stride, name):
        return tf.layers.max_pooling2d(inputs=x,
                                       pool_size=[pool, pool],
                                       strides=stride,
                                       name=name,
                                       data_format='channels_first')

    # ENCODE
    # BLOCK 1
    x = conv(input_layer, 64, 3, 'block1_conv1')
    x = conv(x, 64, 3, 'block1_conv2')
    x = pool(x, 2, 2, 'block1_pool')

    # BLOCK 2
    x = conv(x, 128, 3, 'block2_conv1')
    x = conv(x, 128, 3, 'block2_conv2')
    x = pool(x, 2, 2, 'block2_pool')

    # BLOCK 3
    x = conv(x, 256, 3, 'block3_conv1')
    x = conv(x, 256, 3, 'block3_conv2')
    x = conv(x, 256, 3, 'block3_conv3')
    x = pool(x, 2, 2, 'block3_pool')

    # BLOCK 4
    x = conv(x, 512, 3, 'block4_conv1')
    x = conv(x, 512, 3, 'block4_conv2')
    x = conv(x, 512, 3, 'block4_conv3')
    x = pool(x, 2, 2, 'block4_pool')

    # BLOCK 5
    x = conv(x, 512, 3, 'block5_conv1')
    x = conv(x, 512, 3, 'block5_conv2')
    x = conv(x, 512, 3, 'block5_conv3')
    x = pool(x, 2, 2, 'block5_pool')

    # DECODE
    # BLOCK 1
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(x)
    x = conv(x, 512, 3, 'block1_deconv1')
    x = conv(x, 512, 3, 'block1_deconv2')
    x = conv(x, 512, 3, 'block1_deconv3')

    # BLOCK 2
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(x)
    x = conv(x, 512, 3, 'block2_deconv1')
    x = conv(x, 512, 3, 'block2_deconv2')
    x = conv(x, 256, 3, 'block2_deconv3')

    # BLOCK 3
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(x)
    x = conv(x, 256, 3, 'block3_deconv1')
    x = conv(x, 256, 3, 'block3_deconv2')
    x = conv(x, 128, 3, 'block3_deconv3')

    # BLOCK 4
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(x)
    x = conv(x, 128, 3, 'block4_deconv1')
    x = conv(x, 64, 3, 'block4_deconv2')

    # BLOCK 5
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_first")(x)
    x = conv(x, 64, 3, 'block5_deconv1')
    x = tf.layers.conv2d(inputs=x,
                         filters=NUM_CLASSES,
                         kernel_size=1,
                         padding='same',
                         name='block5_deconv2',
                         data_format='channels_first')

    logits = tf.transpose(x, perm=[0, 2, 3, 1])

    # x = tf.reshape(x, [-1, x.shape[1], x.shape[2] * x.shape[3]])
    # x = tf.transpose(x, perm=[0, 2, 1])
    # logits = tf.nn.softmax(x, name='softmax')
    # prediction = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CLASSES])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.nn.softmax(logits, name='softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # labels = tf.reshape(labels, [-1, labels.shape[1] * labels.shape[2]])
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)


def main(unused_argv):
    train_data = util.load_train_img(tiling=False)
    train_labels = util.load_train_lbl(tiling=False)
    predict_data = util.load_test_data(tiling=False)

    train_labels = np.around(train_labels)
    train_labels = train_labels.astype('int32')

    # EXPAND to 608 x 608
    train_data = np.pad(train_data, ((0, 0), (104, 104), (104, 104), (0, 0)), 'reflect')
    train_labels = np.pad(train_labels, ((0, 0), (104, 104), (104, 104)), 'reflect')

    # Channel first
    # train_data = np.rollaxis(train_data, -1, 1)
    # predict_data = np.rollaxis(predict_data, -1, 1)

    # neeed to expand the channel axis for the image augmentation
    train_labels = np.expand_dims(train_labels, 3)

    # Create the Estimator
    road_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="outputs/road")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=constants.BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    road_estimator.train(
        input_fn=train_input_fn,
        max_steps=(constants.N_SAMPLES * constants.NUM_EPOCH) // constants.BATCH_SIZE)

    # road_estimator.train(
    #     input_fn=train_input_fn,
    #     max_steps=10)

    # Predicions
    # Do prediction on test data
    util.create_prediction_dir("predictions_test/")
    file_names = util.get_file_names()

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        num_epochs=1,
        shuffle=False,
        batch_size=constants.BATCH_SIZE)

    predictions = road_estimator.predict(input_fn=predict_input_fn)
    res = [p['classes'] for p in predictions]

    for i in range(constants.N_TEST_SAMPLES):
        labels = res[i]
        img = util.label_to_img_full(IMG_SIZE, IMG_SIZE, labels)
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_test/' + file_names[i])

    # Do prediction on train data
    util.create_prediction_dir("predictions_train/")

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        num_epochs=1,
        shuffle=False,
        batch_size=constants.BATCH_SIZE)

    predictions = road_estimator.predict(input_fn=predict_input_fn)
    res = [p['classes'] for p in predictions]

    for i in range(constants.N_SAMPLES):
        labels = res[i]
        img = util.label_to_img_full(IMG_SIZE, IMG_SIZE, labels)
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_train/satImage_{:03}.png'.format(i + 1))


if __name__ == "__main__":
    tf.app.run()
