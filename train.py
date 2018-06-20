import numpy
from cnn_model import CnnModel

EPOCH_COUNT = 40

model = CnnModel()
numpy.random.seed(42)  # Ensure reproducibility
model.model.summary()
model.train(EPOCH_COUNT)
model.save('model_weights.h5')
