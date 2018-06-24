#!/usr/bin/env python

from mickey import *

EPOCH_COUNT = 40

# model = Inceptuous("Inceptuous-opt-test")
# model = InceptionResNet("Inception-ResNet-v2")
model = ResNet("ResNet", full_preactivation=False)
# model = MyModel("NasNetLarge")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
