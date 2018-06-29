#!/usr/bin/env python

from mickey import *
from helpers import *
import datetime

d = datetime.datetime.utcnow()  # <-- get time in UTC
timestamp = d.isoformat("T") + "Z"
predict_on = "test_images"

#model = Inceptuous("Inceptuous-opt-test")
#model = InceptionResNet()
#model = ResNet(full_preactivation=False)
model = RedNet("RedNet", full_preactivation=False)
#model.load('model-weights_2018-06-28T22:26:12.354756Z.h5')
model.model.summary()

submission_filename = 'submission-' + timestamp + '_' + model.MODEL_NAME + '.csv'
submission_directory = 'prediction-' + timestamp + '_' + model.MODEL_NAME

post_processing = True
generate_submission_heatmaps(model, os.path.join("data", predict_on), submission_directory, post_processing)
#generate_submission(model, os.path.join("data", predict_on), submission_filename, post_processing)
#generate_overlay_images(model, os.path.join("data", predict_on), post_processing)
