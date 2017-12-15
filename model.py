# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.feature
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

r = 0.4     #scale down
width = 100 #patch size

def make_patches(img, res, h1, w1):
    trainX = []
    trainY = []
    
    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i,j,:])
            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])
            
    return trainX, trainY

def GetData(image_1, image_2):
    
    img1 = cv2.GaussianBlur(image_1, (5,5),0)
    
#    plt.figure(figsize=(50, 50))
#    plt.imshow(image_1)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    
    

    #this image_4 contains only dots in the image    
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)


##    cv2.imshow('image',image_4)
##    cv2.waitKey(0)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4,axis=2)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h,w,d = image_2.shape

    res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')
    
    for blob in blobs:
        
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25: # RED
            res[x1,y1,0]+=1
        elif R > 225 and b > 225 and g < 25: # MAGENTA
            res[x1,y1,1]+=1
        elif R < 75 and b < 50 and 150 < g < 200: # GREEN
            res[x1,y1,4]+=1
        elif R < 75 and  150 < b < 200 and g < 75: # BLUE
            res[x1,y1,3]+=1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1,y1,2]+=1

    ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    
    img = cv2.resize(image_2 * ma, (int(w*r),int(h*r)))
    
    dotted_img = cv2.resize(image_1 * ma, (int(w*r),int(h*r)))
    
#    ma2 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
#    img2 = cv2.resize(image_2 * ma2, (int(w*r),int(h*r)))
    
    h1,w1,d = img.shape
    
    h2, w2, d2 = dotted_img.shape
    
    trainX, trainY = make_patches(img, res, h1, w1)
    
    dotted_trainX, dotted_trainY = make_patches(dotted_img, res, h1, w1)

#    for i in range(int(w1//width)):
#        for j in range(int(h1//width)):
#            trainY.append(res[i,j,:])
#            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])
#            
#    for i in range(int(w2//width)):
#        for j in range(int(h2//width)):
#            dotted_trainY.append(res[i, j,:])
#            dotted_trainX.append(dotted_img[(j * width): (j * width + width), (i * width + width), :])
    
    return np.array(trainX), np.array(trainY), np.array(dotted_trainX), np.array(dotted_trainY)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print ('hahahaha')

# read the Train and Train Dotted images
filename = '0.jpg'
    
dotted_image = cv2.imread("D:/ML work/NOAA Sea Lion count/Data/TrainDotted/" + filename)
image = cv2.imread("D:/ML work/NOAA Sea Lion count/Data/Train/" + filename)

trainX, trainY, dotted_trainX, dotted_trainY = GetData(dotted_image, image)

np.random.seed(1004)
randomize = np.arange(len(trainX))
np.random.shuffle(randomize)
trainX = trainX[randomize]
trainY = trainY[randomize]

n_train = int(len(trainX) * 0.7)
testX = trainX[n_train:]
testY = trainY[n_train:]
trainX = trainX[:n_train]
trainY = trainY[:n_train]

fig = plt.figure(figsize=(50, 50))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(trainX[i], cv2.COLOR_BGR2RGB))
    
print(trainY[:4])

np.random.seed(1004)
randomize = np.arange(len(dotted_trainX))
np.random.shuffle(randomize)
dotted_trainX = dotted_trainX[randomize]
dotted_trainY = dotted_trainY[randomize]

n_train2 = int(len(dotted_trainX) * 0.7)
testX = dotted_trainX[n_train2:]
testY = dotted_trainY[n_train2:]
dotted_trainX = dotted_trainX[:n_train2]
dotted_trainY = dotted_trainY[:n_train2]

fig = plt.figure(figsize=(50, 50))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(dotted_trainX[i], cv2.COLOR_BGR2RGB))
    
print(dotted_trainY[:4])

trainX, trainY, dotted_trainX, dotted_trainY = GetData(cv2.cvtColor(dotted_trainX[0], cv2.COLOR_BGR2RGB), 
                                                                    cv2.cvtColor(trainX[0], cv2.COLOR_BGR2RGB),)

np.random.seed(1004)
randomize = np.arange(len(trainX))
np.random.shuffle(randomize)
trainX = trainX[randomize]
trainY = trainY[randomize]

n_train = int(len(trainX) * 0.7)
testX = trainX[n_train:]
testY = trainY[n_train:]
trainX = trainX[:n_train]
trainY = trainY[:n_train]

fig = plt.figure(figsize=(50, 50))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(trainX[i], cv2.COLOR_BGR2RGB))
    
print(trainY[:4])

np.random.seed(1004)
randomize = np.arange(len(dotted_trainX))
np.random.shuffle(randomize)
dotted_trainX = dotted_trainX[randomize]
dotted_trainY = dotted_trainY[randomize]

n_train2 = int(len(dotted_trainX) * 0.7)
testX = dotted_trainX[n_train2:]
testY = dotted_trainY[n_train2:]
dotted_trainX = dotted_trainX[:n_train2]
dotted_trainY = dotted_trainY[:n_train2]

fig = plt.figure(figsize=(50, 50))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(dotted_trainX[i], cv2.COLOR_BGR2RGB))
    
print(dotted_trainY[:4])

#hsv = cv2.cvtColor(dotted_trainX[0], cv2.COLOR_BGR2HSV)
#
#img1 = cv2.GaussianBlur(hsv, (5,5),0)
#
#lower_red = np.array([30,150,50])
#upper_red = np.array([255,255,180])
#
#mask = cv2.inRange(img1, lower_red, upper_red)
#res = cv2.bitwise_and(dotted_trainX[0],dotted_trainX[0], mask= mask)
#
#edges = cv2.Canny(img1,600,900)
#
#cv2.imshow('Edges',edges)
#cv2.waitKey(0)






#model = Sequential()
#
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(width,width,3)))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dense(5, activation='linear'))
#
#initial_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(300,300,3))
#last = initial_model.output
#x = Flatten()(last)
#x = Dense(1024)(x)
#x = LeakyReLU(alpha=.1)(x)
#preds = Dense(5, activation='linear')(x)
#model = Model(initial_model.input, preds)
#
#optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
#model.compile(loss='mean_squared_error', optimizer=optim)
#model.fit(trainX, trainY, epochs=8, verbose=2)
