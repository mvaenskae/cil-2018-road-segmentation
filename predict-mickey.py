#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow() # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

# model = Inceptuous("Inceptuous-opt-test")
# model = InceptionResNet("Inception-ResNet-v2")
model = ResNet("ResNet", full_preactivation=False)
model.load('model_weights.h5')
model.model.summary()


submission_filename = 'submission-' + timestamp + '_' + predict_on + '.csv'

predict_on_images(os.path.join("data", predict_on), model, submission_filename)