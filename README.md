# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[border]: ./examples/border.png
[angles]: ./examples/angles.png
[mse]: ./examples/mse.png
[modifieddata]: ./examples/modifieddata.png
[originalhistogram]: ./examples/original_histogram.png
[modifiedhistogram]: ./examples/modified_histogram.png
[video]: ./video.mp4

Overview
---
This write-up was written as a partial fulfillment of the requirements for the Nano degree of "Self-driving car engineer" at the Udacity. The goal of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The project instructions and starter code can be download [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3).
The project environment can be created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md).

Rubric Points
---
This write-up explains the points in rubric by providing the description in each step and links to other supporting documents and the images to demonstrate how the code works with examples.
### Required File Submission
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
All required files are submitted in the directory containing this write-up.
The files are
* `model.py` : a python script which setups the architecture model, generates training data, and trains the model
* `model.h5` : a trained model file
* `drive.py` : a python scrip connecting the model to the simulator
* `video.mp4` : a video file capturing simulation during autonomous driving mode
* this write-up file

### Quality of Code
#### 1. Submission includes functional code
The code is fully functional to successfully drive the car within the Udacity simulator framework. This can be verified by executing `python drive.py model.h5`. The simulation was recorded in the `video.mp4` file play.

#### 2. Submission code is usable and readable
The code in `model.py` uses a Python generator which is indispensable for this kind of project. Otherwise, the process for training quickly runs out of memory when just trying to read in all the image data. The task in `model.py` code is structured with functions and commented where needed to render codes readable.

### Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed
The used model is based on NVIDIA's [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper. This model is slightly modified to fit the input image size.
#### 2. Attempts to reduce overfitting in the model
In order to overcome the side-effect of model-overfitting, Drop-out layers are introduced where needed. All collected data are also split into training and validation data with 8-to-2 ratio. The model is finally tested by running it through the simulator.
#### 3. Model parameter tuning
The model parameters are tuned with *Adam* optimizer with the epoch number of 10.
#### 4. Appropriate training data
Training data are collected by driving the car in the center of track. Then, the data are populated in the preprocessing function in `model.py`. The manual recording of track-recovery data is the error-prone manipulation for a clumsy driver Instead, the image preprocessing techniques generates the necessary recovery data.
For details about how the training data are augmented, see the next section.

### Architecture and Training Documentation
#### 1. Solution Design Approach
The key part of this project is to secure high quality of training data. The approach to this end is as follows. First, the training data are collected by running simulator. (A training datum consists of both the image and steering angle.)
Then, several image processing techniques are employed to augment the training data.
The 20% of training data are reserved for a validation data set and the other data are for a training data set.
The model based on [NVIDIA's paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) is trained with these augmented data.
The final step is to run the simulator to see how well the car drives around track.
At the end of some parameter tuning, the car in the simulator is able to drive around the track without leaving the road in the autonomous mode.

#### 2. Final Model Architecture
The employed model architecture is summarized in the below.
Since the input image size is (160, 320, 3), the first layer cropped the input image to (60, 180, 3). This layer not only extract the only useful region of interest but also reduces the number of hyper-parameters. The second layer normalizes the data between -1 and 1 for the prevention of over-fitting. 
From the third layer, the neural network processing begins. The model consists of 5 convolution neural network with the following layer size and flatten layer with decreasing neurons.
The total number of parameters amounts to 1,426,039.

|Layer (type)                    |Output Shape         |Param #    |Features                        |
|--------------------------------|---------------------|-----------|--------------------------------|
|cropping2d_1 (Cropping2D)       |(None, 60, 180, 3)   |0          |crop (160,320,3)-->(60,180,3)   |
|lambda_1 (Lambda)               |(None, 60, 180, 3)   |0          |normailzation function call     |
|convolution2d_1 (Conv2D)        |(None, 28, 88, 24)   |1824       |24(5x5) with relu               |
|convolution2d_2 (Conv2D)        |(None, 12, 42, 36)   |21636      |36(5x5) with relu               |
|convolution2d_3 (Conv2D)        |(None, 4, 19, 48)    |43248      |48(5x5) with relu               |
|convolution2d_4 (Conv2D)        |(None, 2, 17, 64)    |27712      |64(3x3) with relu               |
|convolution2d_5 (Conv2D)        |(None, 1, 16, 64)    |16448      |64(2x2) with relu               |
|flatten_1 (Flatten)             |(None, 1024)         |0          |flattened to 1024               |
|dropout_1 (Dropout)             |(None, 1024)         |0          |drop-out proba = 0.5            |
|dense_1 (Dense)                 |(None, 1164)         |1193100    |1164 neurons                    |
|dropout_2 (Dropout)             |(None, 1164)         |0          |drop-out proba = 0.5            |
|dense_2 (Dense)                 |(None, 100)          |116500     |100 neurons                     |
|dropout_3 (Dropout)             |(None, 100)          |0          |drop-out proba = 0.5            |
|dense_3 (Dense)                 |(None, 50)           |5050       |50 neurons                      |
|dropout_4 (Dropout)             |(None, 50)           |0          |drop-out proba = 0.5            |
|dense_4 (Dense)                 |(None, 10)           |510        |10 neurons                      |
|dense_5 (Dense)                 |(None, 1)            |11         |1 neurons                       |

#### 3. Creation of the Training Set & Training Process
First of all, the collected training data distribution is evaluated in the blow. As it is seen, almost all data are concentrated around the origin. It is because the car is driven in the center of the track during a training mode driving. This is a good driving habit but these data does not teach the neural network how to recover the car when it approaches to the border of the track.

![originalhistogram]

To augment the training data, the following techniques are devised. The following figure is an example of the collected image from the center. The portion in the rectangular window in blue is extracted from the image and only used for a training. In this way, the parameters to be trained can be considerably reduced. Besides, this does not deteriorate the training effect because the useful steering angle information is concentrated in this blue window.

![border]

Then, this blue window is allowed to move around at random within the red line borders in the image. This enables the car to simulate in different road conditions such as up/down hills and curved roads.
In case of left/right movement of the window, the steering angle is corrected by 0.008 x number of pixel movement in side. The following figure shows this effect with adjusted angle.

![angles]

When generating the training data in `model.py`, This extracted blue window is pasted to the original image in the center on purpose. Since the first layer in the neural network crops the input image to get exactly this blue window, only this extracted window will be presented to the neural network for a training. When tested in the simulation, the network will also see the only centered view which is what is intended.
In addition, the brightness of the blue window portion is also adjusted for a training data diversity and flipped image is also used. However, the right/left camera images are not used in this project because it is considered to be redundant with the above data augmentation technique. If the right/left camera images are used, the correction term should be carefully chosen to be consistent with the other augmented data.

![modifieddata]

The following histogram shows the distribution of training angle data augmented by the techniques explained in the above. Surely, this get more uniform angle data distribution which is a desirable property of a training data set.

![modifiedhistogram]

The deep neural network is trained with the epoch number of 10. Each epoch presents 12856 training data to the neural network. The mean squared error loss curve is shown in the below.

![mse]

### Simulation
The car is able to navigate correctly on the track which is shown in `video.mp4` file play. No tire deviate the drive portion of the track surface.
![video]
