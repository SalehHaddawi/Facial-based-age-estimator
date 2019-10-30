from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Test GPU
# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None)



# Init CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(16 , 3 , 3 ,input_shape= (16,16,3), activation = 'relu'))

#Step 2 - MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# #To Increase accuracy
classifier.add(Convolution2D(16 , 3 , 3 , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# # #To Increase accuracy
# classifier.add(Convolution2D(64 , 3 , 3 , activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))

# # To Increase accuracy
# classifier.add(Convolution2D(64 , 3 , 3 , activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))

# # To Increase accuracy
# classifier.add(Convolution2D(64 , 3 , 3 , activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connented --> Neural Network
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(6, activation='sigmoid'))



#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




#Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../datasets/ages/training',
        target_size=(16, 16),
        batch_size=8,
        class_mode='categorical')

test_Set = test_datagen.flow_from_directory(
        '../datasets/ages/test',
        target_size=(16, 16),
        batch_size=8,
        class_mode='categorical')

#Set Callback
callback = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')

classifier.fit_generator(
        training_set,
        steps_per_epoch=43943,
        epochs=1,
        validation_data=test_Set,
        validation_steps=10779)

print('Save ...')
classifier.save('h.hdf5')
print('Saved to  : h.hdf5')


