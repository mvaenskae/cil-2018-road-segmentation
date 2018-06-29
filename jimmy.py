from cnn_model import *
from keras.layers import Activation, Concatenate

class VEGGG(CnnModel):
    def __init__(self, model_name):
        super().__init__()
        self.MODEL_NAME = model_name

    def build_model(self):
        layers = BasicLayers(relu_version='parametric')

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = layers.cbr(x, 32, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers.cbr(x, 64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')

        x2 = layers._flatten(x)
        x3 = layers._dense(x2, 2 * ((self.CONTEXT * self.CONTEXT) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x3 = layers._dropout(x3, rate=0.5)
        x3 = layers._act_fun(x3)
        x4 = layers._dense(x3, self.NB_CLASSES)
        x5 = Activation('softmax')(x4)
        self.model = Model(inputs=input_tensor, outputs=x5)

class Simple(CnnModel):
    def __init__(self, model_name):
        super().__init__()
        self.MODEL_NAME = model_name

    def build_model(self):
        layers = BasicLayers(relu_version='leaky')

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = layers.cbr(x, 64, kernel_size=(5, 5), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._spatialdropout(x)

        x = layers.cbr(x, 128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._spatialdropout(x)

        x = layers.cbr(x, 256, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._spatialdropout(x)

        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._spatialdropout(x)

        x = layers.cbr(x, 512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same')
        x = layers._max_pool(x, pool=(2, 2), strides=(2, 2), padding='same')
        x = layers._spatialdropout(x)

        x2 = layers._flatten(x)
        x3 = layers._dense(x2, 2 * 256)
        x3 = layers._dropout(x3, rate=0.5)
        x3 = layers._act_fun(x3)
        x4 = layers._dense(x3, self.NB_CLASSES)
        x5 = Activation('softmax')(x4)
        self.model = Model(inputs=input_tensor, outputs=x5)


