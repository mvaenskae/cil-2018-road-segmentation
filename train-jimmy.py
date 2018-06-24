#!/usr/bin/env python

from jimmy import *

EPOCH_COUNT = 1

model = VEGGG("VEGGG")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
