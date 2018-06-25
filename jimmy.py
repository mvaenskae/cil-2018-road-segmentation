from cnn_model import *
from keras.layers import Activation, Concatenate

class Alexa(CnnModel):
    def __init__(self, model_name):
        super().__init__()
        self.MODEL_NAME = model_name

    def build_model(self):
        layers = BasicLayers(relu_version='parametric')

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = layers._conv2d(x, 96, kernel_size=(11, 11), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._batch_norm(x)

        x = layers._conv2d(x, 256, kernel_size=(5, 5), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._batch_norm(x)

        x = layers._conv2d(x, 384, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')

        x = layers._conv2d(x, 384, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')

        x = layers._conv2d(x, 256, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        x2 = layers._flatten(x)

        x3 = layers._dense(x2, 4096)
        x3 = layers._dropout(x3, rate=0.5)
        x3 = layers._act_fun(x3)

        x4 = layers._dense(x3, 4096)
        x4 = layers._dropout(x4, rate=0.5)
        x4 = layers._act_fun(x4)

        x5 = layers._dense(x4, self.NB_CLASSES)
        x5 = layers._dropout(x5, rate=0.5)
        x5 = layers._act_fun(x5)

        out = Activation('softmax')(x5)
        self.model = Model(inputs=input_tensor, outputs=out)


