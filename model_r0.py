import csv
import cv2
import sklearn
import math
import numpy as np
import os
import os.path
import shutil
import scipy
from scipy import ndimage
import random

samples = []
d = os.getcwd()
#with open(d+'/data/driving_log.csv') as csvfile:
with open(d+'/data_sample/data/driving_log_comb_1.csv') as csvfile:
#with open(d+'/data_sample/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                name = d+'/data_sample/data/IMG/'+batch_sample[0].split('/')[-1]
                #name = d+'/data/IMG/'+batch_sample[0].split('/')[-1]
                image = ndimage.imread(name)
                measurement = float(batch_sample[3])
                images.append(image)
                measurements.append(measurement)

            augmented_images,augmented_measurements = [],[]
            for image, measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

#images = []
#measurements = []
#for line in lines:
#    for i in range(3):
#        source_path = line[i]
#        filename = source_path.split('/')[-1]
#        current_path = d+'/data/IMG/' + filename
        #image = cv2.imread(current_path)
        #image = ndimage.imread(current_path)
        #images.append(image)
        #measurement = float(line[3])
        #measurements.append(measurement)
    
#augmented_images,augmented_measurements = [],[]
#for image, measurement in zip(images,measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)
    #pbar.update(i)
#pbar.finish()
#print
    
    
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)
    
# X_train = np.array(images)
#y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
# from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import BatchNormalization, Dropout

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(BatchNormalization()) # overfitting
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
#model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
#model.add(BatchNormalization()) # overfitting
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
#model.add(BatchNormalization()) # overfitting
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
#model.add(BatchNormalization()) # overfitting
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))

#model.add(Conv2D(6,(5,5),activation="relu"))
#model.add(MaxPooling2D())
#model.add(Conv2D(6,(5,5),activation="relu"))
#model.add(MaxPooling2D())

model.add(Flatten())

from keras.regularizers import l1 # overfitting

model.add(Dense(100))
#model.add(Dense(100),kernel_regularizer=l1(0.001)) # overfitting
model.add(Dropout(0.2)) # overfitting
model.add(Dense(50))
#model.add(Dense(50),kernel_regularizer=l1(0.001)) # overfitting
model.add(Dropout(0.2)) # overfitting
model.add(Dense(10))
#model.add(Dense(10),kernel_regularizer=l1(0.001)) # overfitting
model.add(Dropout(0.2)) # overfitting
model.add(Dense(1))
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train,validation_split = 0.2,shuffle=True,epochs = 2)
history_object = model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)          
model.save('model.h5')

from keras.models import Model
import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

#plt.plot([1,2,3,4])

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('test1.png')
#plt.show()
        
