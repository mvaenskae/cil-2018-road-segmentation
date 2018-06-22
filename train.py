#!/usr/bin/env python

from resnet import ResNet

EPOCH_COUNT = 40

model = ResNet("ResNet", full_preactivation=False)
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
