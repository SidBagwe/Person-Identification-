

import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import cv2
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# number of possible label values
nb_classes = 62

# Initialising the CNN
#callback = [EarlyStopping(monitor='loss', patience=0, verbose=1)]
model = Sequential()


# 1st Convolution Layer
model.add(Conv2D(64,(3,3), padding='same', input_shape=(64, 64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 5th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 6th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening
model.add(Flatten())


# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

print(model.summary())

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


(images, lables, names, id) = ([], [], {}, 0)
os.chdir("/content/data") 
datasets = 'train'
for (subdirs, dirs, files) in os.walk(datasets): 
    for subdir in dirs: 
        names[id] = subdir 
        subjectpath = os.path.join(datasets, subdir) 
        for filename in os.listdir(subjectpath): 
            path = subjectpath + '/' + filename 
            lable = id
            img=image.load_img(path,target_size=(64,64,3))
            img=image.img_to_array(img)
            img=img/255
            images.append(img)
            lables.append(int(lable)) 
        id += 1
X=np.array(images)
print(X.shape)
  
# Create a Numpy array from the two lists above 
#(images, lables) = [np.array(lis) for lis in [images, lables]] 



(images1, lables1, names1, id) = ([], [], {}, 0)
os.chdir("/content/data") 
datasets = 'test'
for (subdirs, dirs, files) in os.walk(datasets): 
    for subdir in dirs: 
        names1[id] = subdir 
        subjectpath = os.path.join(datasets, subdir) 
        for filename in os.listdir(subjectpath): 
            path = subjectpath + '/' + filename 
            lable1 = id
            img1=image.load_img(path,target_size=(64,64,3))
            img1=image.img_to_array(img1)
            img1=img1/255
            images1.append(img1)
            
            lables1.append(int(lable1))
        id += 1
Y=np.array(images1)
print(Y.shape)       
# Create a Numpy array from the two lists above 
#(images1, lables1) = [np.array(lis) for lis in [images1, lables1]]         


model.fit(X,lables,
 epochs = 500,
 validation_data = (Y,lables1),
 batch_size = 64)



model.save('face_62_500.h5')
