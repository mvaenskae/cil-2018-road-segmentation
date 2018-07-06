from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import constants
import tensorflow as tf
import util

tf.logging.set_verbosity(tf.logging.INFO)

weight_decay = 0.0001


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
    if mode == tf.estimator.ModeKeys.TRAIN:
        # FLIP UP DOWN
        flip_ud = lambda x: tf.image.random_flip_up_down(x)
        input_layer = tf.map_fn(flip_ud, input_layer)

        # FLIP LEFT RIGHT
        flip_lr = lambda x: tf.image.flip_left_right(x)
        input_layer = tf.map_fn(flip_lr, input_layer)

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

        tf.summary.image('Augmentation', input_layer, max_outputs=16)

    def dense_layer(x, units, name):
        return tf.layers.dense(x, units=units,
                               kernel_initializer=tf.keras.initializers.he_normal(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                               name=name)

    def bn_relu(x):
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        return x

    def conv(x, out_filters, k_size, name):
        return tf.layers.conv2d(x, filters=out_filters,
                                kernel_size=k_size,
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name=name)

    def pool(x, size, stride, name):
        return tf.layers.max_pooling2d(x,
                                       (size, size),
                                       (stride, stride),
                                       name=name)

    # BLOCK 1
    x = conv(input_layer, 64, 3, 'block1_conv1')
    x = bn_relu(x)
    x = conv(x, 64, 3, 'block1_conv2')
    x = bn_relu(x)
    x = pool(x, 2, 2, 'block1_pool')

    # BLOCK 2
    x = conv(x, 128, 3, 'block2_conv1')
    x = bn_relu(x)
    x = conv(x, 128, 3, 'block2_conv2')
    x = bn_relu(x)
    x = pool(x, 2, 2, 'block2_pool')

    # BLOCK 3
    x = conv(x, 256, 3, 'block3_conv1')
    x = bn_relu(x)
    x = conv(x, 256, 3, 'block3_conv2')
    x = bn_relu(x)
    x = conv(x, 256, 3, 'block3_conv3')
    x = bn_relu(x)
    x = conv(x, 256, 3, 'block3_conv4')
    x = bn_relu(x)
    x = pool(x, 2, 2, 'block3_pool')

    # BLOCK 4
    x = conv(x, 512, 3, 'block4_conv1')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block4_conv2')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block4_conv3')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block4_conv4')
    x = bn_relu(x)
    x = pool(x, 2, 2, 'block4_pool')

    # BLOCK 5
    x = conv(x, 512, 3, 'block5_conv1')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block5_conv2')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block5_conv3')
    x = bn_relu(x)
    x = conv(x, 512, 3, 'block5_conv4')
    x = bn_relu(x)
    x = pool(x, 2, 2, 'block5_pool')

    # FC Layer
    x_shape = x.get_shape().as_list()
    x = tf.reshape(x, [-1, x_shape[1] * x_shape[2] * x_shape[3]])

    x = dense_layer(x, 4096, 'fc1')
    x = bn_relu(x)
    x = tf.layers.dropout(inputs=x, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    x = dense_layer(x, 4096, 'fc2')
    x = bn_relu(x)
    x = tf.layers.dropout(inputs=x, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=x, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Cann add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # load provided images
    train_data = util.load_train_img(tiling=False)
    train_labels = util.load_train_lbl(tiling=True)
    predict_data = util.load_test_data(tiling=False)
    train_labels = util.one_hot_to_num(train_labels)
    # expansion
    train_data = util.crete_patches_large(train_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING, is_mask=False)
    predict_data = util.crete_patches_large(predict_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING,
                                            is_mask=False)

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

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = road_estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Do prediction on test data
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        num_epochs=1,
        shuffle=False)

    predictions = road_estimator.predict(input_fn=predict_input_fn)
    res = [p['probabilities'] for p in predictions]

    file_names = util.get_file_names()
    util.create_prediction_dir("predictions_test/")
    offset = 1444

    for i in range(1, constants.N_TEST_SAMPLES + 1):
        img = util.label_to_img_inverse(608, 608, 16, 16, res[(i - 1) * offset:i * offset])
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_test/' + file_names[i - 1])

    # Predictions Train
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        num_epochs=1,
        shuffle=False)

    predictions = road_estimator.predict(input_fn=predict_input_fn)

    res = [p['probabilities'] for p in predictions]
    util.create_prediction_dir("predictions_train/")
    for i in range(1, 101):
        img = util.label_to_img_inverse(400, 400, constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE,
                                        res[(i - 1) * 625:i * 625])
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_train/{:03}.png'.format(i))


if __name__ == "__main__":
    tf.app.run()
