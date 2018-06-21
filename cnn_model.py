# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.layers import Input, Add
from keras.optimizers import Adam, Nadam, SGD
from keras import losses
from keras import metrics
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import Sequence
from helpers import *

resnet_features = [64, 128, 256, 512]
resnet_repetitions_small = [2, 2, 2, 2]
resnet_repetitions_normal = [3, 4, 6, 3]
resnet_repetitions_large = [3, 4, 23, 3]
resnet_repetitions_extra = [3, 8, 36, 3]

class CnnModel:
    FULL_PREACTIVATION = True
    RELU_VERSION = 'parametric'
    LEAKY_RELU_ALPHA = 0.01
    DATA_FORMAT = 'channels_first'

    def __init__(self):
        """ Construct a CNN classifier. """
        self.patch_size = 16
        self.context = 80
        self.padding = (self.context - self.patch_size) // 2
        self.model = None
        self.initialize()

    def initialize(self):
        """ Initialize or reset this model. """
        patch_size = self.patch_size
        window_size = self.context
        padding = self.padding
        nb_classes = 2
        RESNET_FEATURES = resnet_features
        RESNET_REPETITIONS = resnet_repetitions_normal

        # Compatibility with Theano and Tensorflow ordering
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            input_shape = (3, window_size, window_size)
        else:
            input_shape = (window_size, window_size, 3)

        def _conv2d(_input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same'):
            return Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=CnnModel.DATA_FORMAT,
                dilation_rate=dilation_rate,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )(_input)

        def _batch_norm(_input, dense):
            # https://github.com/keras-team/keras/issues/1921#issuecomment-193837813
            return BatchNormalization(
                axis=1 if not dense else -1,  # Normalize over last axis in dense, else channel-wise for _conv2d
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones',
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None
            )(_input)

        def _act_fun(_input, relu_version=CnnModel.RELU_VERSION):
            if relu_version == 'leaky':
                return LeakyReLU(alpha=CnnModel.LEAKY_RELU_ALPHA)(_input)
            elif relu_version == 'parametric':
                return PReLU(
                    alpha_initializer='zeros',
                    alpha_regularizer=None,
                    alpha_constraint=None,
                    shared_axes=None  # Use channel-wise which is reportedly giving better performance than being shared
                )(_input)
            else:
                return ReLU()(_input)

        def _max_pool(_input, pool=(2, 2), strides=(2, 2), padding='same'):
            return MaxPooling2D(
                pool_size=pool,
                strides=strides,
                padding=padding,
                data_format=CnnModel.DATA_FORMAT
            )(_input)

        def _dense(_input, neurons):
            return Dense(
                units=neurons,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )(_input)

        def _flatten(_input):
            return Flatten(data_format=CnnModel.DATA_FORMAT)(_input)

        def _cbr(_input, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), padding='same', get_model=False):
            x_orig = _input
            x = x_orig
            if not self.FULL_PREACTIVATION:
                x = _conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
            x = _batch_norm(x, False)
            x = _act_fun(x)
            if self.FULL_PREACTIVATION:
                x = _conv2d(x, filters, kernel_size, strides, dilation_rate, padding)
            if get_model:
                Model(input=x_orig, output=x)
            else:
                return x

        def resnet_stem(_input):
            x = _input
            x = _conv2d(x, filters=64, kernel_size=(5, 5), strides=(2, 2))
            x = _batch_norm(x)
            x = _act_fun(x)
            x = _max_pool(x, pool=(3, 3))
            return x

        def _vanilla_branch(_input, filters, strides, dilation=(1, 1)):
            x = _input
            x = _cbr(x, filters=filters, kernel_size=(3, 3), strides=strides, dilation_rate=dilation)
            x = _cbr(x, filters=filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation)
            return x

        def _bottleneck_branch(_input, filters, strides, dilation=(1, 1)):
            x = _input
            x = _cbr(x, filters=filters, kernel_size=(1, 1), strides=strides, dilation_rate=dilation)
            x = _cbr(x, filters=filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation)
            x = _cbr(x, filters=4*filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=dilation)
            return x

        def _shortcut(_input, filters, strides=(1, 1), is_bottleneck=False):
            x = _input
            first_filters = filters
            if is_bottleneck:
                first_filters = 4 * filters
            x = _cbr(x, first_filters, kernel_size=(1, 1), strides=strides)
            return x

        def resnet_vanilla(_input, filters, is_first=False):
            if is_first:
                if filters == 64:
                    strides = 1
                else:
                    strides = 2
                shortcut = _shortcut(_input, filters, strides)
                residual = _vanilla_branch(_input, filters, strides)
            else:
                shortcut = _input
                residual = _vanilla_branch(_input, filters, strides=(1, 1))
            res = Add()([shortcut, residual])
            return res

        def resnet_bottleneck(_input, filters, is_first=False):
            if is_first:
                if filters == 64:
                    strides = 1
                else:
                    strides = 2
                shortcut = _shortcut(_input, filters, strides, True)
                residual = _bottleneck_branch(_input, filters, strides)
            else:
                shortcut = _input
                residual = _bottleneck_branch(_input, filters, strides=(1, 1))
            res = Add()([shortcut, residual])
            return res

        input_tensor = Input(shape=input_shape)
        x = input_tensor
        x = resnet_stem(x)
        for i, layers in enumerate(RESNET_REPETITIONS):
            for j in range(layers):
                x = resnet_vanilla(x, RESNET_FEATURES[i], (j == 0))

        x = _flatten(x)
        x = _dense(x, 2 * ((self.context * self.context) // (self.patch_size * self.patch_size)))
        x = _act_fun(x)
        x = _dense(x, nb_classes)
        x = Activation('softmax')(x)
        self.model = Model(input=input_tensor, outputs=x)

    def train(self, epochs):
        """
        Train this model with the given dataset.
        """
        seed = 42
        np.random.seed(seed)
        batch_size = 16
        nb_classes = 2
        nb_epoch = epochs

        # Read images from disk
        images, groundtruth = read_images_plus_labels()
        samples_per_image = images.shape[1] * images.shape[2] // (self.patch_size * self.patch_size)

        # Pad images (by appling mirror boundary conditions)
        images_padded = np.empty((images.shape[0],
                         images.shape[1] + 2 * self.padding, images.shape[2] + 2 * self.padding,
                         images.shape[3]))
        ground_truth_padded = np.empty((groundtruth.shape[0],
                         groundtruth.shape[1] + 2 * self.padding, groundtruth.shape[2] + 2 * self.padding))

        for i in range(images.shape[0]):
            images_padded[i] = pad_image(images[i], self.padding)
            ground_truth_padded[i] = pad_image(groundtruth[i], self.padding)

        # Generate training and validation set
        images_train, groundtruth_train, images_validate, groundtruth_validate = split_dataset(images_padded, ground_truth_padded, seed)

        batches_train = (images_train.shape[0] * samples_per_image) // batch_size
        batches_validate = (images_validate.shape[0] * samples_per_image) // batch_size

        print('Dataset shape:', images.shape, '( Train:', images_train.shape[0], '| Validate:', images_validate.shape[0], ')')
        print('Samples per image:', samples_per_image, '( Trainsteps per epoch:', batches_train, '| Validatesteps per epoch:', batches_validate, ')')

        def batch_generator(__img, __gt):
            """
            Batch generator which returns `batch_size` many elements in a tuple.
            Designed to run in parallel to generate a better performing pipeline.
            """
            while True:
                # Generate one minibatch
                X_batch = np.empty((batch_size, self.context, self.context, 3))
                Y_batch = np.empty((batch_size, 2))
                for i in range(batch_size):
                    # Select a random image
                    idx = np.random.choice(__img.shape[0])
                    shape = __img[idx].shape

                    # Sample a random window from the image
                    center = np.random.randint(self.context // 2, shape[0] - self.context // 2, 2)
                    sub_image = __img[idx][center[0] - self.context // 2:center[0] + self.context // 2,
                                center[1] - self.context // 2:center[1] + self.context // 2]
                    gt_sub_image = __gt[idx][
                                   center[0] - self.patch_size // 2:center[0] + self.patch_size // 2,
                                   center[1] - self.patch_size // 2:center[1] + self.patch_size // 2]

                    # Image augmentation
                    # Random flip
                    if np.random.choice(2) == 0:
                        # Flip vertically
                        sub_image = np.flipud(sub_image)
                    if np.random.choice(2) == 0:
                        # Flip horizontally
                        sub_image = np.fliplr(sub_image)

                    # Random rotation in steps of 90°
                    num_rot = np.random.choice(4)
                    sub_image = np.rot90(sub_image, num_rot)

                    # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
                    label = (np.array([np.mean(gt_sub_image)]) > 0.25) * 1
                    label = np_utils.to_categorical(label, nb_classes)
                    X_batch[i] = sub_image
                    Y_batch[i] = label

                if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
                    X_batch = np.rollaxis(X_batch, 3, 1)

                yield (X_batch, Y_batch)

        validation_data = ValidationSequence(images_validate, groundtruth_validate, batch_size, nb_classes,
                                             self.context, self.patch_size, batches_validate)

        validate_patches = []
        validate_labels = []
        for i in range(batches_validate):
            temp_patches, temp_labels = next(validation_data)
            validate_patches.append(temp_patches)
            validate_labels.append(temp_labels)
        validate_patches = np.rollaxis(np.reshape(np.asarray(validate_patches), (batches_validate * batch_size, self.context, self.context, 3)), 3, 1)
        validate_labels = np.reshape(np.asarray(validate_labels), (batches_validate * batch_size, 2))
        validate_data = (validate_patches, validate_labels)

        # Reduce learning rate iff validation accuracy not improving
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=5, verbose=1,
                                                 mode='auto', min_delta=1e-3, cooldown=0, min_lr=1e-8)

        # Stop training early iff validation accuracy not improving
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', min_delta=1e-3, patience=11, verbose=1,
                                       mode='auto')

        # Enable Tensorboard logging with graphs, gradients, images and historgrams
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  batch_size=batch_size,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # Hacky Tensorboard wrapper
        tensorboard_hack = TensorBoardWrapper(validate_data,
                                              nb_steps=batches_validate,
                                              log_dir='./logs',
                                              histogram_freq=1,
                                              batch_size=batch_size,
                                              write_graph=True,
                                              write_grads=False,
                                              write_images=False)  # Visualizations of layers where applicable, not superbly useful

        # Save the model's state on each epoch, given the epoch has better fitness
        filepath = "weights-{epoch:03d}-{val_binary_accuracy:.4f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor='val_binary_accuracy',
                                       verbose=1,
                                       save_best_only=False,
                                       period=1)

        # The following functions have been copied from
        # https://github.com/keras-team/keras/issues/5400#issuecomment-314747992
        def mcor(y_true, y_pred):
            # matthews_correlation
            y_pred_pos = K.round(K.clip(y_pred, 0, 1))
            y_pred_neg = 1 - y_pred_pos

            y_pos = K.round(K.clip(y_true, 0, 1))
            y_neg = 1 - y_pos

            tp = K.sum(y_pos * y_pred_pos)
            tn = K.sum(y_neg * y_pred_neg)

            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)

            numerator = (tp * tn - fp * fn)
            denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            return numerator / (denominator + K.epsilon())

        def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.

                Only computes a batch-wise average of recall.

                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                """Precision metric.

                Only computes a batch-wise average of precision.

                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        self.model.compile(loss=losses.binary_crossentropy,
                           optimizer=Adam(lr=1e-4),
                           metrics=[metrics.binary_accuracy, mcor, f1])

        try:
            self.model.fit_generator(
                generator=batch_generator(images_train, groundtruth_train),
                steps_per_epoch=batches_train,
                epochs=nb_epoch,
                verbose=1,
                # callbacks=[tensorboard_hack, checkpointer, reduce_lr_on_plateau, early_stopping],
                callbacks=[tensorboard, checkpointer, reduce_lr_on_plateau, early_stopping],
                # validation_data=validate_data,
                validation_data=batch_generator(images_validate, groundtruth_validate),
                validation_steps=batches_validate,
                use_multiprocessing=True,
                shuffle=True)
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')

    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)

    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)

    def classify(self, X):
        """
        Classify an unseen set of samples.
        This method must be called after "train".
        Returns a list of predictions.
        """
        # Subdivide the images into blocks
        img_patches = create_patches(X, self.patch_size, 16, self.padding)

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            img_patches = np.rollaxis(img_patches, 3, 1)

        # Run prediction
        Z = self.model.predict(img_patches)
        Z = (Z[:,0] < Z[:,1]) * 1

        # Regroup patches into images
        return group_patches(Z, X.shape[0])


class ValidationSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, classes, context_size, patch_size, limit):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.classes = classes
        self.context = context_size
        self.patch_size = patch_size
        self.limit = limit
        self.idx = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        X_batch = np.empty((self.batch_size, self.context, self.context, 3))
        Y_batch = np.empty((self.batch_size, 2))

        for i in range(self.batch_size):
            # Select a random image
            idx = np.random.choice(self.x.shape[0])
            shape = self.x[idx].shape

            # Sample a random window from the image
            center = np.random.randint(self.context // 2, shape[0] - self.context // 2, 2)
            sub_image = self.x[idx][center[0] - self.context // 2:center[0] + self.context // 2,
                        center[1] - self.context // 2:center[1] + self.context // 2]
            gt_sub_image = self.y[idx][
                           center[0] - self.patch_size // 2:center[0] + self.patch_size // 2,
                           center[1] - self.patch_size // 2:center[1] + self.patch_size // 2]

            # Image augmentation
            # Random flip
            if np.random.choice(2) == 0:
                # Flip vertically
                sub_image = np.flipud(sub_image)
            if np.random.choice(2) == 0:
                # Flip horizontally
                sub_image = np.fliplr(sub_image)

            # Random rotation in steps of 90°
            num_rot = np.random.choice(4)
            sub_image = np.rot90(sub_image, num_rot)

            # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
            label = (np.array([np.mean(gt_sub_image)]) > 0.25) * 1
            label = np_utils.to_categorical(label, 2)
            X_batch[i] = sub_image
            Y_batch[i] = label

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            X_batch = np.rollaxis(X_batch, 3, 1)

        return (X_batch, Y_batch)

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx > self.limit:
            self.idx = 0
            raise StopIteration
        else:
            return self.__getitem__(self.idx - 1)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''
    # https://github.com/keras-team/keras/issues/3358#issuecomment-312531958

    def __init__(self, val_data, batch_size, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.val_data = val_data      # Validation data of (patches, labels)
        self.batch_size = batch_size  # Size of single batch
        self.nb_steps = nb_steps      # Number of batches

    def on_epoch_end(self, epoch, logs):
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib = self.val_data[0][s * self.batch_size: (s + 1) * self.batch_size]
            tb = self.val_data[1][s * self.batch_size: (s + 1) * self.batch_size]
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.float32)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)
