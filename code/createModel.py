"""
Creating and training of the model
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print(tf.config.experimental.list_physical_devices('GPU'))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout
from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy,binary_accuracy
from tensorflow.keras import regularizers
import tensorflow.keras
# import tensorflow as tf
import os




# from keras.callbacks import ModelCheckpoint
# from keras.models import Model, load_model, save_model, Sequential
# from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
# from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
# from keras.optimizers import Adam
# # from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

# Init CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(96 , (7 , 7) ,input_shape= (128,128,1), activation = 'relu', padding='same', kernel_initializer='random_uniform'))
# classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(3,3)))
classifier.add(BatchNormalization())


# # #To Increase accuracy
classifier.add(Conv2D(96 , (5 , 5) , activation = 'relu', use_bias=True))
# classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(3,3)))
classifier.add(BatchNormalization()) 
# classifier.add(tf.compat.v2.nn.local_response_normalization())
# classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, (3 , 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3,3)))

# classifier.add(Conv2D(64, (3 , 3), padding='same', activation='relu'))

# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(Dropout(0.25))



# classifier.add(Conv2D(384  , (3 , 3) , activation = 'relu', use_bias=True))
# # classifier.add(Dropout(0.2))
# classifier.add(MaxPooling2D(pool_size=(3,3)))
# classifier.add(BatchNormalization()   ) 
# classifier.add(tf.compat.v2.nn.local_response_normalization())


# classifier.add(Conv2D(256 , (3 , 3) , activation = 'relu', bias_initializer=initializers.Constant(0.1)))
# # classifier.add(Dropout(0.2))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(BatchNormalization()) 

# classifier.add(Conv2D(256 , (3 , 3) , activation = 'relu', bias_initializer=initializers.Constant(0.1)))
# # classifier.add(Dropout(0.2))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(BatchNormalization()) 

# classifier.add(Conv2D(128 , (3 , 3) , activation = 'relu', bias_initializer=initializers.Constant(0.1)))
# classifier.add(Dropout(0.5))
# classifier.add(MaxPooling2D(pool_size=(3,3)))

# classifier.add(Conv2D(256 , (3 , 3) , activation = 'relu', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
# classifier.add(Dropout(0.5))
# classifier.add(MaxPooling2D(pool_size=(3,3)))

# classifier.add(Conv2D(256 , (3 , 3) , activation = 'relu'))
# classifier.add(Dropout(0.1))

# classifier.add(MaxPooling2D(pool_size=(2,2)))

# classifier.add(Conv2D(64 , (3 , 3) , activation = 'relu'))
# classifier.add(Dropout(0.25))
# classifier.add(MaxPooling2D(pool_size=(1,1)))

# # #To Increase accuracy
# classifier.add(Conv2D(64 ,( 3 , 3 ), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# classifier.add(Dropout(0.5))

# # # #To Increase accuracy
# classifier.add(Conv2D(128 ,( 3 , 3 ), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(7,7)))
# classifier.add(Dropout(0.5))

# # # #To Increase accuracy
# classifier.add(Conv2D(32 ,( 3 , 3 ), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connented --> Neural Network
# classifier.add(Dense(128,activation='relu', bias_initializer=initializers.Constant(0.1)))
# classifier.add(Dropout(0.5))

# classifier.add(Dense(512,activation='relu', bias_initializer='zeros', kernel_initializer='random_uniform'))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
# classifier.add(BatchNormalization()) 

# classifier.add(Dense(512,activation='relu', bias_initializer='zeros', kernel_initializer='random_uniform'))
# classifier.add(Dropout(0.50))

# classifier.add(Dense(512,activation='relu'))
# classifier.add(Dropout(0.25)) 
# classifier.add(BatchNormalization()) 

# classifier.add(Dense(32,activation='relu', bias_initializer=initializers.Constant(0.1)))
# classifier.add(Dropout(0.5)) 

#Total Classes 
total = len(os.listdir('../datasets/ages/training'))
# , bias_initializer=initializers.Constant(0.1)
classifier.add(Dense(total, activation='softmax'))

# opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
opt = SGD(lr=0.001)

# optimazer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

#Compiling CNN
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

checkpoint = ModelCheckpoint("../../model/ep22/model{epoch:08d}_{val_loss:.2f}_{val_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1,
    save_best_only=False, mode='auto', period=1)

#Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        # height_shift_range=0.2,
        horizontal_flip=False
        )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../datasets/ages/training',
        target_size=(128, 128),
        # batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

print("#############")

print(training_set)
print(type(training_set))

print("#############")



test_Set = test_datagen.flow_from_directory(
        '../datasets/ages/test',
        target_size=(128, 128),
        # batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

#Set Callback   
callback = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')
classifier.fit_generator(
        training_set,
        steps_per_epoch=12432,
        epochs=60,
        validation_data=test_Set,
        validation_steps=3103,
        workers=250,
        callbacks=[checkpoint]
        )

# print('Save ...')
# classifier.save('model/n.hdf5')
# print('Saved to  : Model/n.hdf5')


