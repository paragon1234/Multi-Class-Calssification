# Multi-Class-Calssification

## Goal

The goals / steps of this project are the following:
* Build, a convolution neural network in Keras that performs 3 class classification with one class as none
* Train and validate the model with a training and validation set
* Predict the class label of the test data.
* Create an output csv file that contains the image filename of test data and the predicted class


## Files Submitted & Code Quality

### 1. Submission includes all required files

My project includes the following files:
* 'Multi Calss Calssification.py' containing the python-Keras code to create, train and test the model
* output.csv file containing the image filename and the predicted calss

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Overview

### 1. An appropriate model architecture has been employed

My model consists of a 3 convolution neural network with 3x3 filter sizes and depths between 32 and 128 (line 32-46) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras batchNorm

The image of calss 'aircraft' and 'none' are of 20x20 pixels, while that of class 'ship' are 80x80 pixels. The input to the CNN is 20x20 pixels.

### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting by train_validation_split (line 65). To reduce overfitting, early stopping is introduced thrrough callbacks (line 93-97). The model was tested on test dataset and the accuracy on test dataset is 100%.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

### 4. Appropriate training data

Training/Test data is provided in the file "gnr638-mls4rs-a1.zip"

I randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was determined through early stopping using validation accuracy as the indicator.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use different layers so that vehicle stays on track without touching the side lines as if it is safe for the person sitting inside the car while doing actual driving.

My first step was to use only a fully connected layer(FCL) and check out the result. It was found that when the vehicle moved on a bridge, whose texture was different from the track, then it hit the side walls.

Then I improved the model by adding a convolution neural network layer(CNN). Using CNN the vehicle successfully traversed the bridge, but it was not able to take sharp turn.

The next improvement was to use 2 layers of CNN with relu, and 3 FCL. In this case the vehicle was able to slightly turn steering angle on sharp turn, but it still went off the track.

Then taking clue from Nvidia architecture I created a trimmed down architecture: 3 CNN (output depth of 24, 36, 64) with relu units, followed by 4 FCL. In this case the vehicle was safely able to take sharp turn, but it moved from one side of the track to another, like a drunken car.

My final archtitecture was the Nvidia Architecture, where the car movement from one side of track to another reduced. At the end of the process, the vehicle is able to drive autonomously around the track (at 25 speed) without leaving the track.

### 2. Final Model Architecture

The final model architecture (model.py lines 54-63) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					            | 
|:-----------------:|:---------------------------------------------:| 
| Input         		| 20x20x3 color image   						            | 
| Convolution 3x3   | same padding, output 20x20x32 	              |
| RELU					    |												                        |
| Convolution 3x3   | 2x2 stride, same padding, output 10x20x64 	  |
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







