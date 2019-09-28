# importing libraries 
import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K 
  
  
img_width, img_height = 20, 20
  
train_data_dir =  '/content/drive/My Drive/KaggleData_split/train'
test_data_dir  =  '/content/drive/My Drive/KaggleData_split/test'
nb_train_samples = 1500 
nb_test_samples = 120
epochs = 100
batch_size = 16
num_of_classes=3 #change it as per your number of categories
validation_split = 0.2
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
  
# Model  
model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape = input_shape)) 
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Conv2D(64, (3, 3))) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3))) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2)))
model.add(Dropout(0.25))
  
model.add(Flatten()) 
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25)) 
model.add(Dense(num_of_classes, activation='softmax'))

  
model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
 
#Train/Validation/Test Generators
train_datagen = ImageDataGenerator( rescale = 1. / 255,
                                  validation_split=validation_split) # set validation split)

test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), shuffle=True, 
                     batch_size = batch_size, class_mode ='categorical',
                                 subset='training') # set as training data 

validation_generator = train_datagen.flow_from_directory( train_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical',
                         subset='validation') # set as validation data 

test_generator = test_datagen.flow_from_directory( test_data_dir, 
                   target_size =(img_width, img_height), shuffle=False,
          batch_size = 1, class_mode ='categorical') 


# confirm the iterator works
#batchX, batchy = train_generator.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#batchX, batchy = validation_generator.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#batchX, batchy = test_generator.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
  
model.fit_generator(train_generator, 
    steps_per_epoch = train_generator.samples // batch_size, callbacks=[es, mc], 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size)
"""
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size) 
"""

# load the saved model
saved_model = load_model('best_model.h5')

# make a prediction
test_generator.reset()
yhat = saved_model.predict_generator(test_generator, steps= nb_test_samples)
print(len(yhat))
  
#model.save_weights('model_saved.h5')

###
# Save to csv file
###
import os
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import csv

#sort files in test directory
l=os.listdir('/content/drive/My Drive/KaggleData/test/test')
li=[x.split('.')[0] for x in l]
li.sort()

#Create final labels
labels = numpy.argmax(yhat, axis=1)
#print(labels)
temp = numpy.empty([nb_test_samples, 2], dtype=int)

for i in range(nb_test_samples):
    if labels[i]==1:
        labels[i]=3
    if labels[i]==0:
        labels[i]=1
    temp[i] = [li[i], labels[i]]

numpy.savetxt('/content/drive/My Drive/194310001.csv', temp, fmt='%d',delimiter=',', header='imageId,label')


#Test Generator Debugging Technique
#print(test_generator.classes)
#print(test_generator.class_indices)
#print(test_generator.filenames)
#test_generator.reset()
#image,label = test_generator.next()
#import matplotlib.pyplot as plt
#plt.imshow(image.reshape(20,20,3))
#plt.show()
