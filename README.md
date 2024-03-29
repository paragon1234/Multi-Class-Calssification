# Multi-Class-Classification

## Goal

The goals / steps of this project are the following:
* Build, a convolution neural network in Keras that performs 3 class classification with one class as none
* Train and validate the model with a training and validation set
* Predict the class label of the test dataset.
* Create an output csv file that contains the image filename of test data and the predicted class


## Files Submitted

My project includes the following files:
* 'Multi Class Classification.py' containing the python-Keras code to create, train and test the model
* output.csv file containing the image filename and the predicted class

The "Multi Class Classification.py" file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Overview

### 1. An appropriate model architecture has been employed

My model consists of a 3 convolution neural network with 3x3 filter sizes and depths between 32 and 128 (line 32-46) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using the Keras batchNorm layer

The image size of class 'aircraft' and 'none' is 20x20 pixels, while that of class 'ship' is 80x80 pixels. The input to the CNN is an image of size 20x20 pixels.

### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (achieved through train_validation_split: line 65). To reduce overfitting, early stopping is introduced through callbacks (line 93-97). The model was tested on test dataset and the accuracy on test dataset is 100%.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 61).

### 4. Appropriate training data

Training/Test data is provided in the file "gnr638-mls4rs-a1.zip"

I randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine, if the model was over or under fitting. The ideal number of epochs is achieved through early stopping using validation accuracy as the indicator.

## Model Architecture and Training Strategy

### 1. Important things to change/check in multiclass (compared to binary classification):

Set class_mode='categorical' in the generator() function(s).

Don't forget that the last dense layer must specify the number of labels (or classes):

model.add(layers.Dense(3, activation='softmax'))

Make sure that activation='softmax' and loss='categorical_crossentropy' is chosen- to suit multiclass problems

### 2. Order of test data for prediction
* It is important to specify shuffle=False in test_datagen.flow_from_directory(), in order to preserve the order of filenames and predictions.
* If a generator has been set up before, it may be necessary to rest it using test_generator.reset(), else weird outputs will come.
* Change the value of batch_size in test_datagen.flow_from_directory() from default value (which is batch_size=32 ) to batch_size=1. Then set the steps of predict_generator to the total number of your test images.

### 3. Solution Design Approach

The overall strategy for deriving a model architecture was to use different layers and then find the accuracy on the test dataset.

I started with a model with 3 Convolution layers and 3 fully connected layers. It also incorporated batch normalization, dropout, early stopping and 30% train- validation split. The model gave 100% accuracy on test dataset.

I then changed train-validation split to 20%. Now the accuracy on test dataset reduced to 99.1%

I then changed train-validation split back to 30%, and removed batchNormalization and droupout layers. The test accuracy was still 100% .

I then reduced the Convolution layers to 2 and fully connected layers also to 2 (without batchNorm, droupout and 30% test-validation split). It still result in 100% test accuracy

My final archtitecture was with 3 Convolution layers and 3 fully connected layers.. At the end, the test accuracy was 100%.

### 4. Final Model Architecture

The final model architecture (lines 31-57) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					            | 
|:-----------------:|:---------------------------------------------:| 
| Input         		| 20x20x3 color image   						            | 
| Convolution 3x3   | same padding, output 20x20x32 	              |
| RELU					    |												                        |
| Convolution 3x3   | 2x2 stride, same padding, output 10x10x64 	  |
| RELU					    |												                        |
| Convolution 3x3		| 2x2 stride, same padding, output 5x5x128      |
| RELU					    |												                        |
| Flatten				    | outputs 3200 								                  |
| Fully connected		| Output 512        							              |
| RELU					    |												                        |
| Fully connected		| Output 64 									                  |
| RELU					    |												                        |
| Fully connected		| Output 3                                      |
| RELU					    |												                        |
|						        |												                        |







