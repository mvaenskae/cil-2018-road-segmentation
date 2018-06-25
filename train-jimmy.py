#!/usr/bin/env python

from jimmy import *

EPOCH_COUNT = 40

model = Alexa("Alexa")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('alexa_weights.h5')
