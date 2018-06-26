#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow() # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

# model = Inceptuous("Inceptuous-opt-test")
model = InceptionResNet()
# model = ResNet("ResNet", full_preactivation=False)
model.load('model_weights.h5')
# model.load('weights-Inception-ResNet-v2-e014-f1-0.6475.hdf5')
model.model.summary()


submission_filename = 'submission-' + timestamp + '_' + predict_on + '.csv'

post_processing = False
generate_submission(model, os.path.join("data", predict_on), submission_filename, post_processing)
generate_overlay_images(model, os.path.join("data", predict_on), post_processing)
