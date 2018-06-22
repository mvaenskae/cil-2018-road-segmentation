# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, Nadam, SGD
from keras import losses
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from helpers import *
from keras_helpers import *

class CnnModel:
    """
    Base class for any CNN model.
    """
    def __init__(self, patch_size=16, context=96, data_format='channels_first', relu_version=None, leaky_relu_alpha=0.01):
        """ Construct a CNN classifier. """
        self.PATCH_SIZE = patch_size
        self.CONTEXT = context
        self.PADDING = (self.CONTEXT - self.PATCH_SIZE) // 2
        self.NB_CLASSES = 2
        self.BATCH_SIZE = 16
        self.DATA_FORMAT = data_format
        self.RELU_VERSION = relu_version
        self.LEAKY_RELU_ALPHA = leaky_relu_alpha
        self.model = None
        self.MODEL_NAME = 'Base'

        # Compatibility with Theano and Tensorflow ordering
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            self.INPUT_SHAPE = (3, self.CONTEXT, self.CONTEXT)
        else:
            self.INPUT_SHAPE = (self.CONTEXT, self.CONTEXT, 3)

        self.build_model()
    
    def build_model(self):
        """
        This method is required to be overwritten by a child-class and supposed to generate the network.
        :return: Nothing
        """
        raise NotImplementedError('CnnModel::build_model is not yet implemented.')

    def train(self, epochs):
        """
        Train this model with the given dataset.
        """
        seed = 42
        np.random.seed(seed)
        nb_epoch = epochs

        # Read images from disk
        images, groundtruth = read_images_plus_labels()
        samples_per_image = images.shape[1] * images.shape[2] // (self.PATCH_SIZE * self.PATCH_SIZE)

        # # Pad images (Apply mirror boundary conditions)
        # images_padded = np.empty((images.shape[0],
        #                           images.shape[1] + 2 * self.PADDING, images.shape[2] + 2 * self.PADDING,
        #                           images.shape[3]))
        # groundtruth_padded = np.empty((groundtruth.shape[0],
        #                                 groundtruth.shape[1] + 2 * self.PADDING, groundtruth.shape[2] + 2 * self.PADDING))
        #
        # for i in range(images.shape[0]):
        #     images_padded[i] = pad_image(images[i], self.PADDING)
        #     groundtruth_padded[i] = pad_image(groundtruth[i], self.PADDING)

        # Generate training and validation set
        images_train, groundtruth_train, images_validate, groundtruth_validate = split_dataset(images, groundtruth, seed)

        batches_train = (images_train.shape[0] * samples_per_image) // self.BATCH_SIZE
        batches_validate = (images_validate.shape[0] * samples_per_image) // self.BATCH_SIZE

        print('Dataset shape:', images.shape, '( Train:', images_train.shape[0], '| Validate:', images_validate.shape[0], ')')
        print('Samples per image:', samples_per_image, '( Trainsteps per epoch:', batches_train, '| Validatesteps per epoch:', batches_validate, ')')

        validation_data = ImageSequence(images_validate, groundtruth_validate, images_validate, groundtruth_validate,
                                        self.BATCH_SIZE, self.NB_CLASSES, self.CONTEXT, self.PATCH_SIZE,
                                        batches_validate)
        training_data = ImageSequence(images_train, groundtruth_train, images_train, groundtruth_train, self.BATCH_SIZE,
                                      self.NB_CLASSES, self.CONTEXT, self.PATCH_SIZE, batches_train)

        # Reduce learning rate iff validation accuracy not improving
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=5, verbose=1,
                                                 mode='auto', min_delta=1e-3, cooldown=0, min_lr=1e-8)

        # Stop training early iff validation accuracy not improving
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=1e-3, patience=11, verbose=1,
                                       mode='auto')

        # Enable Tensorboard logging with graphs, gradients, images and historgrams
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  batch_size=self.BATCH_SIZE,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # Hacky Tensorboard wrapper
        # tensorboard_hack = TensorBoardWrapper(validate_data,
        #                                       nb_steps=batches_validate,
        #                                       log_dir='./logs',
        #                                       histogram_freq=1,
        #                                       batch_size=batch_size,
        #                                       write_graph=True,
        #                                       write_grads=False,
        #                                       write_images=False)  # Visualizations of layers where applicable, not superbly useful

        # Save the model's state on each epoch, given the epoch has better fitness
        filepath = "weights-" + self.MODEL_NAME + "-{epoch:03d}-{val_categorical_accuracy:.4f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor='val_categorical_accuracy',
                                       verbose=1,
                                       save_best_only=False,
                                       period=1)

        # Shuffle/augment images at the start of each epoch
        image_shuffler = ImageShuffler(validation_data, training_data)

        def softmax_crossentropy_with_logits(y_true, y_pred):
            """
            Applies the loss function as found in the template not using logits (so use softmax layer at the end)
            :param y_true: Ground-truth values
            :param y_pred: Network predictions
            :return: Application of K.categorical_crossentropy()
            """
            return K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=1)

        self.model.compile(loss=softmax_crossentropy_with_logits,
                           optimizer=Nadam(lr=1e-4),
                           metrics=[ExtraMetrics.mcor, ExtraMetrics.cil_error, metrics.categorical_accuracy])

        try:
            self.model.fit_generator(
                generator=training_data,
                steps_per_epoch=batches_train,
                epochs=nb_epoch,
                verbose=1,
                # callbacks=[tensorboard_hack, checkpointer, reduce_lr_on_plateau, early_stopping],
                callbacks=[tensorboard, checkpointer, reduce_lr_on_plateau, early_stopping, image_shuffler],
                # validation_data=validate_data,
                validation_data=validation_data,
                validation_steps=batches_validate,
                # shuffle=True,  #Not needed, our generator shuffles everything already
                use_multiprocessing=False)
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
        Classify a set of samples. This method should be called after successful training and loading of the model.
        :param X: Full-size image which to classify.
        :return: List of predictions.
        """
        # Subdivide the images into blocks
        img_patches = create_patches(X, self.PATCH_SIZE, 16, self.PADDING)

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            img_patches = np.rollaxis(img_patches, 3, 1)

        # Run prediction
        Z = self.model.predict(img_patches)

        Z = (Z[:, 0] < Z[:, 1]) * 1

        # Regroup patches into images
        return group_patches(Z, X.shape[0])
