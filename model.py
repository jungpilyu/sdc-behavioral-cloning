# -*- coding: utf-8 -*-
"""
CarND Project 3
Behaviroral Clonning
December 4th, 2017
Jungpil YU
"""

import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

def get_model():
    '''
    return a Neural Network Model based on NVIDIA's paper
    (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    Dropout layers are added to the original model and some convolution2D layer's 
    filter paprameters are changed to fit the new input image format
    '''
    model = Sequential()
    model.add(Cropping2D(cropping=((top_crop+v_limit,bottom_crop+v_limit), (h_limit,h_limit)), input_shape = (row, col, ch)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,2,2,activation="relu"))  # change : 64,3,3 --> 64,2,2
    model.add(Flatten())
    model.add(Dropout(0.5))                             # added layer
    model.add(Dense(1164))
    model.add(Dropout(0.5))                             # added layer
    model.add(Dense(100))
    model.add(Dropout(0.5))                             # added layer
    model.add(Dense(50))
    model.add(Dropout(0.5))                             # added layer
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def adjust_gamma(image):
    """
    adjust the brightness of training images.
    reference --> http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    argument : input image
    return : Gamma-adjusted image
    """
    invGamma = 1.0 / np.random.uniform(0.4, 1.5)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)     # apply gamma correction using the lookup table

def preprocess_image(img, angle, row_shift = 1.0, col_shift = 1.0):
    """
    generate a modified version of a training image
    arguments :
     - img : the original image
     - angle : the original steering angle
     - row_shift : the amount of row shift (0 < row_shift < 1)
     - col_shift : the amount of column shift (0 < col_shift < 1)
    return :
     - a training image modified in place
     - a steering angle corrected with col_shift value
    """
    # get the window's top left point
    win_row = top_crop + (int)(2*v_limit*row_shift)
    win_col = (int)(2*h_limit*col_shift)
    # extract the ROI image and randomly change the brightness
    cropped_image = img[win_row:win_row+crop_size[0], win_col: win_col + crop_size[1]]
    cropped_image = adjust_gamma(cropped_image)
    # paste the extracted image in the place where the simulator see
    pt = top_crop + v_limit
    img[pt:pt+crop_size[0], h_limit:h_limit+crop_size[1]] = cropped_image
    # correct the original steering angle by col_shift value
    adjusted_angle = angle -2*h_limit*(col_shift-0.5)*angle_per_pixel
    return img, adjusted_angle

def generator(samples, batch_size=32):
    """
    arguments :
     - samples : a training data set
     - batch_size : the number of generated data for an epoch
    return : (2*batch_size) number of training data per one call
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(1): # i = 1 : only center image will be used
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    ori_image = plt.imread(name)
                    ori_angle = float(batch_sample[3])
                    r_s = np.random.rand()
                    r_c = np.random.rand()
                    an_image, an_angle = preprocess_image(ori_image, ori_angle, r_s, r_c)
                    images.append(an_image)
                    # create adjusted steering measurements for the side camera images
                    # but not used here
                    if i == 1:
                        an_angle = an_angle + correction
                    elif i == 2:
                        an_angle = an_angle - correction
                        
                    angles.append(an_angle)
                    # add a training data by flipping the image
                    flip_img = cv2.flip(an_image,1)
                    images.append(flip_img)
                    angles.append(-an_angle)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# various parameters
correction = 0.2            # steering angle correction for right/left camera image
ch, row, col = 3, 160, 320  # Input image format given by the Udacity simulator
learning_rate = 0.0002      # manual setting of learning rate
h_limit = 70                # horizontal margin of the blue window
v_limit = 10                # vertical margin of the blue window
top_crop = 40               # cropped pixels of the input image from the top
bottom_crop = 40            # cropped pixels of the input image from the bottom
angle_per_pixel = 0.008     # steering angle correction for the blue window movement
crop_size = row -top_crop - bottom_crop - 2*v_limit, col - 2*h_limit

if __name__ == '__main__':

    # get samples from the csv file
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    samples = samples[1:] # eliminate the head row

    # split samples into a training and a validation set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    ### Get the neural network model and set an optimizer
    model = get_model()
    model.compile(optimizer=Adam(learning_rate), loss="mse")

    ### Training and Validating by fit_generator
    history_object = model.fit_generator(train_generator, samples_per_epoch = 2*len(train_samples),
                validation_data = validation_generator, nb_val_samples = 2*len(validation_samples),
                nb_epoch=10)

    ### save model, weight, and optimization status
    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
