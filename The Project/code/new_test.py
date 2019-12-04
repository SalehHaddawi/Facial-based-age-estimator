import scipy.io
from datetime import datetime, timedelta

from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras


def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    exact_date = datetime.fromordinal(int(datenum)) \
                 + timedelta(days=int(days)) + timedelta(hours=int(hours)) \
                 + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) \
                 - timedelta(days=366)

    return exact_date.year


def getImagePixels(image_path):
    img = keras_image.load_img("../datasets/wiki_crop/%s" % image_path[0], grayscale=False, target_size=(128, 128))

    x = keras_image.img_to_array(img).reshape(1, -1)[0]
    # x = preprocess_input(x)
    return x


mat = scipy.io.loadmat('../datasets/wiki_crop/wiki.mat')

instances = mat['wiki'][0][0][0].shape[1]

columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

df = pd.DataFrame(index=range(0, instances), columns=columns)

for i in mat:
    if i == "wiki":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            df[columns[j]] = pd.DataFrame(current_array[j][0])

df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)

df['age'] = df['photo_taken'] - df['date_of_birth']

# remove pictures does not include face
df = df[df['face_score'] != -np.inf]

# some pictures include more than one face, remove them
df = df[df['second_face_score'].isna()]

# check threshold
df = df[df['face_score'] >= 3]

# some records do not have a gender information
df = df[~df['gender'].isna()]

df = df.drop(columns=['name', 'face_score', 'second_face_score', 'date_of_birth', 'face_location'])

# some guys seem to be greater than 100. some of these are paintings. remove these old guys
df = df[df['age'] <= 100]

# some guys seem to be unborn in the data set
df = df[df['age'] > 0]

print("cleaned the data")

df['pixels'] = df['full_path'].apply(getImagePixels)

print("added pixels data")

classes = 101  # 0 to 100
target = df['age'].values
target_classes = keras.utils.to_categorical(target, classes)

features = []

for i in range(0, df.shape[0]):
    features.append(df['pixels'].values[i])

print("pixels as features")

features = np.array(features)
features = features.reshape(features.shape[0], 128, 128, 3)

print("reshape features")

train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)

print("split train data")

print("begin layers")

# VGG-Face model
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(128, 128, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# pre-trained weights of vgg-face model.
# you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
# related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
model.load_weights('vgg_face_weights.h5')

print("loaded module: ", "vgg_face_weights")

for layer in model.layers[:-7]:
    layer.trainable = False

print("CNN")

base_model_output = Sequential()
base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

age_model = Model(inputs=model.input, outputs=base_model_output)

age_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='age_model{epoch:08d}_{val_loss:.2f}_{val_accuracy:.2f}.hdf5'
                               , monitor="val_loss", verbose=1, save_best_only=True, mode='auto')

scores = []
epochs = 250
batch_size = 256

for i in range(epochs):
    print("epoch ", i)

    ix_train = np.random.choice(train_x.shape[0], size=batch_size)

    score = age_model.fit(train_x[ix_train], train_y[ix_train]
                          , epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer])

    scores.append(score)

age_model.evaluate(test_x, test_y, verbose=1)
