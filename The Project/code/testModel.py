from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf

folder_path = '../datasets/ages/test/baby/18b70c6d8ea7fcb51b43a65efc9ced82.jpg'

# path to model
model_path = 'h.hdf5'

model = load_model(model_path)


img = image.load_img(path=folder_path,target_size=(16,16))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

images = np.vstack([img])
img_class = model.predict_classes(images)

print(img_class)




# scores = model.summary()

# print(scores)
# prediction = img_class[0]
# classname = img_class[0]
# score = model.evaluate(img, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))