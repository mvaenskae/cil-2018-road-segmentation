#!/usr/bin/env python

from ivan import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 3000
model_filename = 'model-weights_segnet.h5'

model = SegNet()
model.model.summary()
model.train(EPOCH_COUNT)
model.save(model_filename)
