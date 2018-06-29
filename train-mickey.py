#!/usr/bin/env python

from mickey import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"

EPOCH_COUNT = 3000
model_filename = 'model-weights_' + timestamp + '.h5'

#model = InceptionResNet()
#model = EasyNet()
#model = ResNet(full_preactivation=False)
model = RedNet(full_preactivation=False)
#model = SimpleNet()
#model.load('model-weights_2018-06-28T18:55:29.199897Z.h5')
model.model.summary()
model.train(EPOCH_COUNT)
model.save(model_filename)
