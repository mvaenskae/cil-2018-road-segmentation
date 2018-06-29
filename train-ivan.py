#!/usr/bin/env python

from ivan import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 100
model_filename = 'model-weights_' + timestamp + '.h5'

model = SegNet("SegNet")
model.model.summary()
model.train(EPOCH_COUNT)
model.save(model_filename)
