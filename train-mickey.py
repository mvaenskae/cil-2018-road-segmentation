#!/usr/bin/env python

from mickey import Inceptuous

EPOCH_COUNT = 40

model = Inceptuous("Inceptuous")
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
