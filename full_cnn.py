# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras import metrics
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from helpers import *
from keras_helpers import *
from base_cnn import AbstractCNN

class FullCNN(AbstractCNN):
    """
    Abstract base class for any CNN model.
    """
    def __init__(self,
                 image_size=608,
                 nb_classes=2,
                 batch_size=8,
                 model_name='FullCNN',
                 data_format='channels_first',
                 relu_version=None,
                 leaky_relu_alpha=0.01):
        """ Construct a CNN classifier. """
        super().__init__(image_size, nb_classes, batch_size, model_name, data_format, relu_version, leaky_relu_alpha)

    def build_model(self):
        """
        This method is required to be overwritten by a child-class and supposed to generate the network.
        :return: Nothing
        """
        raise NotImplementedError('FullCNN::build_model is not yet implemented.')

    def preprocessing_train(self):
        """
        General preprocessing steps for correct training.
        :return: Number of steps for training and validation
        """
        # Read images from disk and generate training and validation set
        self.images, self.groundtruth = read_images_plus_labels()
        self.images_train, self.groundtruth_train, self.images_validate, self.groundtruth_validate = split_dataset(self.images, self.groundtruth)

        # Print some extraneous metrics helpful to users of this template
        batches_train = (self.images_train.shape[0] * (608 // self.IMAGE_SIZE)**2) // self.BATCH_SIZE
        batches_validate = (self.images_validate.shape[0] * (608 // self.IMAGE_SIZE)**2) // self.BATCH_SIZE
        print('Dataset shape:', self.images.shape, '( Train:', self.images_train.shape[0], '| Validate:', self.images_validate.shape[0], ')')
        print('Trainsteps per epoch:', batches_train, '| Validatesteps per epoch:', batches_validate)
        return batches_train, batches_validate

    def train(self, epochs, checkpoint=None, init_epoch=0):
        """
        Train this model.
        """
        np.random.seed(42)
        batches_train, batches_validate = self.preprocessing_train()

        # Generators for image sequences which apply Monte Carlo sampling on them
        training_data = ImageSequenceHeatmaps(self.images_train, self.groundtruth_train, self.BATCH_SIZE,
                                              self.IMAGE_SIZE, batches_train)
        validation_data = ImageSequenceHeatmaps(self.images_validate, self.groundtruth_validate, self.BATCH_SIZE,
                                                self.IMAGE_SIZE, batches_validate)

        # Reduce learning rate iff validation average f1 score not improving for AdamOptimizer
        reduce_lr_on_plateau_adam = ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=5,
                                                      verbose=1,
                                                      mode='min',
                                                      min_delta=5e-3,
                                                      cooldown=0,
                                                      min_lr=1e-7)

        # Stop training early iff validation average f1 score not improving for AdamOptimizer
        early_stopping_adam = EarlyStopping(monitor='val_loss',
                                            min_delta=5e-4,
                                            patience=11,
                                            verbose=1,
                                            mode='min')

        # Reduce learning rate iff validation average f1 score not improving for SGD
        reduce_lr_on_plateau_sgd = ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.25,
                                                     patience=5,
                                                     verbose=1,
                                                     mode='min',
                                                     min_delta=1e-4,
                                                     cooldown=0,
                                                     min_lr=1e-8)

        # Stop training early iff validation average f1 score not improving for AdamOptimizer
        early_stopping_sgd = EarlyStopping(monitor='val_loss',
                                           min_delta=1e-4,
                                           patience=11,
                                           verbose=1,
                                           mode='min')

        # Enable Tensorboard logging and show the graph -- Other options not sensible when using Monte Carlo sampling
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  batch_size=self.BATCH_SIZE,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # Hacky Tensorboard wrapper if a fixed validation set is given to model.generator_fit and this wrapper
        # tensorboard_hack = TensorBoardWrapper(validate_data,
        #                                       nb_steps=batches_validate,
        #                                       log_dir='./logs',
        #                                       histogram_freq=1,
        #                                       batch_size=batch_size,
        #                                       write_graph=True,
        #                                       write_grads=False,
        #                                       write_images=False)

        # Save the model's state on each epoch, given the epoch has better fitness
        filepath = "weights-" + self.MODEL_NAME + "-e{epoch:03d}-ce-{val_loss:.4f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor='val_loss',
                                       mode='min',
                                       verbose=1,
                                       save_best_only=False,
                                       period=500)

        # Shuffle/augment images at the start of each epoch
        image_shuffler = ImageShuffler(training_data, validation_data)

        def softmax_crossentropy_with_logits(y_true, y_pred):
            """
            Applies the loss function as found in the template not using logits (so use softmax layer at the end)
            :param y_true: Ground-truth values
            :param y_pred: Network predictions
            :return: Application of K.categorical_crossentropy()
            """
            return K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=1)

        # Define in a list what callbacks and metrics we want included
        model_callbacks_adam = [tensorboard, checkpointer, image_shuffler]
        model_callbacks_sgd = [tensorboard, checkpointer, reduce_lr_on_plateau_sgd, early_stopping_sgd, image_shuffler]
        model_metrics = [metrics.categorical_accuracy]

        if checkpoint is not None:
            self.model = load_model(checkpoint, custom_objects={'softmax_crossentropy_with_logits': softmax_crossentropy_with_logits})
            print('Loaded checkpoint for model to continue training')
        else:
            self.model.compile(loss=softmax_crossentropy_with_logits,
                               optimizer=Adam(lr=1e-5))
                               #metrics=model_metrics))

        try:
            self.model.fit_generator(
                generator=training_data,
                steps_per_epoch=batches_train,
                epochs=epochs,
                initial_epoch=init_epoch,
                verbose=1,
                callbacks=model_callbacks_adam,
                validation_data=validation_data,
                validation_steps=batches_validate,
                shuffle=False,  # Not needed, our generator shuffles everything already
                use_multiprocessing=False)  # This requires a thread-safe generator which we don't have

            # # TODO: Generate callback which makes this double-call to the network not required.
            # self.model.compile(loss=softmax_crossentropy_with_logits,
            #                    optimizer=SGD(lr=1e-4, momentum=0.9, nesterov=False),
            #                    metrics=model_metrics)
            #
            # self.model.fit_generator(
            #     generator=training_data,
            #     steps_per_epoch=batches_train,
            #     epochs=epochs,
            #     verbose=1,
            #     callbacks=model_callbacks_sgd,
            #     validation_data=validation_data,
            #     validation_steps=batches_validate,
            #     shuffle=False,  # Not needed, our generator shuffles everything already
            #     use_multiprocessing=False)  # This requires a thread-safe generator which we don't have
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            filepath = "weights-" + self.MODEL_NAME + "-SIG2.hdf5"
            self.model.save(filepath, overwrite=True, include_optimizer=True)
            pass
        except:
            # Generic case for SIGUSR2. Stop model training and save current state.
            filepath = "weights-" + self.MODEL_NAME + "-SIGUSR2.hdf5"
            self.model.save(filepath, overwrite=True, include_optimizer=True)

        print('Training completed')

    def classify(self, X):
        """
        Classify a set of samples. This method should be called after successful training and loading of the model.
        :param X: Full-size image which to classify.
        :return: List of predictions.
        """
        # Subdivide the images into blocks
        img_patches = generate_blocks(X, self.IMAGE_SIZE)

        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'tf':
            img_patches = np.rollaxis(img_patches, 3, 1)

        # Run prediction
        Z = self.model.predict(img_patches)
        Z = (np.greater_equal(Z[:, 0], Z[:, 1]) * 255.0).astype('uint8')

        # Regroup patches into images
        return group_blocks(Z, X.shape[0])
