from cnn_model import *
from keras.layers import Activation

class ResNet(CnnModel):
    FULL_PREACTIVATION = False

    def __init__(self, model_name, full_preactivation=False):
        super().__init__()
        self.MODEL_NAME = model_name
        self.FULL_PREACTIVATION = full_preactivation

    def build_model(self):
        resnet = ResNetLayers(self.DATA_FORMAT, self.RELU_VERSION, self.LEAKY_RELU_ALPHA, self.FULL_PREACTIVATION)

        input_tensor = Input(shape=self.INPUT_SHAPE)
        x = input_tensor
        x = resnet.stem(x)
        for i, layers in enumerate(resnet.REPETITIONS_SMALL):
            for j in range(layers):
                x = resnet.vanilla(x, resnet.FEATURES[i], (j == 0))

        x = resnet._flatten(x)
        x = resnet._dense(x, 2 * ((self.CONTEXT * self.CONTEXT) // (self.PATCH_SIZE * self.PATCH_SIZE)))
        x = resnet._act_fun(x)
        x = resnet._dense(x, self.NB_CLASSES)  # Returns a logit
        x = Activation('softmax')(x)  # No logit anymore
        self.model = Model(inputs=input_tensor, outputs=x)