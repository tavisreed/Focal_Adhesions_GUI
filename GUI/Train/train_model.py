#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

#Some more things we need to import
#This will be used to navigate the system and grab the needed images
import os
#This is used to create loading graphics
from tqdm import tqdm_notebook, tnrange
#numpy contains a bunch of useful math related functions
import numpy as np
#This is used to plot
import matplotlib.pyplot as plt
#This can be used to help find a pattern
import fnmatch
#This is for preprocessing the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#These are also for image preprocessing
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
#This is just to quickly and easily split the training data into train and validation sets
from sklearn.model_selection import train_test_split
#Used to find the boundaries within the masks so that a weight map can be genearted
from skimage.segmentation import find_boundaries
from skimage.measure import label
from scipy import ndimage
#For randomness
import random
#For the sigmoid function: math.exp(-x)
import math
import PIL
#To read and display images
import cv2

def train(TrainName, TrainPath, numChannles,tempImageReplaceType,
          tempMaskReplaceType, ImageWidthHeight, MinPercentWhite,
          ValidationSplit, tempSmallKernal, tempSmallBatch,
          tempSmallEpochs, tempSmallThresh, tempBoundaryKernal,
          tempW0, tempSigma, tempBoundaryBatch, tempBoundaryEpochs,
          tempBoundaryThresh, tempMixerKernal, tempMixerBatch,
          tempMixerEpochs):
    print('Starting Training')
    #Some important parameters that will be used later
    #Image Related
    imgWidth=ImageWidthHeight
    imgHeight=ImageWidthHeight
    numChannles=numChannles
    pathTrain='..\\GUI\\Train\\'+TrainPath+'\\'

    global imageReplaceType, imageType, maskReplaceType

    imageReplaceType = '.'+tempImageReplaceType.replace('.', '')
    imageType= '*'+imageReplaceType
    maskReplaceType = '.' + tempMaskReplaceType.replace('.', '')

    #model save name
    global modelSmallSaveName, modelBoundarySaveName
    global NewBoundarySaveName, modelMixerSaveName

    #Check to see if folder for Model alreadt exists. If not, create it
    if not os.path.exists('..\\GUI\\Segment\\Saved_Models\\'+TrainName):
        os.makedirs('..\\GUI\\Segment\\Saved_Models\\'+TrainName)

    modelSmallSaveName='..\\GUI\\Segment\\Saved_Models\\'+TrainName+'\\small.h5'
    modelBoundarySaveName='..\\GUI\\Segment\\Saved_Models\\'+TrainName+'\\boundary.h5'
    NewBoundarySaveName='..\\GUI\\Segment\\Saved_Models\\'+TrainName+'\\newboundary.h5'
    modelMixerSaveName='..\\GUI\\Segment\\Saved_Models\\'+TrainName+'\\mixer.h5'

    MinPercentWhite=MinPercentWhite
    ValidationSplit=ValidationSplit

    #Set up global variables for training
    global SmallKernal, SmallBatch, SmallEpochs, SmallThresh
    global BoundaryKernal, w0, sigma, BoundaryBatch, BoundaryEpochs, BoundaryThresh
    global MixerKernal, MixerBatch, MixerEpochs, MixerThresh
    SmallKernal = tempSmallKernal
    SmallBatch = tempSmallBatch
    SmallEpochs = tempSmallEpochs
    SmallThresh = tempSmallThresh
    BoundaryKernal = tempBoundaryKernal
    w0 = tempW0
    sigma = tempSigma
    BoundaryBatch = tempBoundaryBatch
    BoundaryEpochs = tempBoundaryEpochs
    BoundaryThresh = tempBoundaryThresh
    MixerKernal = tempMixerKernal
    MixerBatch = tempMixerBatch
    MixerEpochs = tempMixerEpochs

    #Here we have a function that just retrives relevent file names
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(name)
        return result

    #Here we create a function for getting data that can be used to retreive training images, their masks, and test images
    def getData(path, imageType, train=True):
        #Here we are getting the id of all the images stored in the folder
        ids=find(imageType, path+"images")
        #We create an array that will hold the images. dtype=np.float32 sets the data type of the array to a 32 bit float
        X=[]
        if train:
            #We are doing the same thing as above, but this time it is for the masks
            y=[]
        print('Retrieving and resizing images!')

        #Here we are actually getting each image and putting them in their respective array. We use tqdm_notebook to create a loading bar
        for n, tempId in tqdm_notebook(enumerate(ids),total=len(ids)):
            #Before we can load the image, we need to do some quick edits to the ID string to make it easier to get our images
            tempId=tempId.replace(imageReplaceType, '')
            #Load in the images
            tempImage=load_img(path+'images\\'+tempId+imageReplaceType, color_mode = "grayscale")
            #We need to make the Image bigger so we can break it into an integer of croped images
            tempy=img_to_array(tempImage)
            #Crop the images to fit instead of resizing which causes interpolation
            size=np.shape(tempy)
            width=size[0]
            height=size[1]
            target_width=0
            target_height=0
            #Determine which dimension is greater
            if height>width:
                while target_height<height:
                    target_height=target_height+imgHeight
                    target_width=target_height
            else:
                while target_height<width:
                    target_width=target_width+imgWidth
                    target_height=target_width
            #Anti-Crop the image
            x_img=tempImage.crop((0,0,target_height,target_width))
            x_img=img_to_array(x_img)
            numRows=round((target_height/imgHeight))
            numCols=round((target_width/imgWidth))
            croppedImgs=[]
            for i in range(numRows):
                for j in range (numCols):
                    tempy=x_img
                    x1=imgWidth*(i)
                    x2=imgWidth*(i+1)
                    y1=imgHeight*(j)
                    y2=imgHeight*(j+1)
                    tempy=tempy[x1:x2, y1:y2, :]
                    croppedImgs.append(tempy)

            #If we are using a training set then we also need to load in the masks
            if train:
                tempMask=load_img(path+'masks\\'+tempId+maskReplaceType)
                tempy=img_to_array(tempMask)
                #Crop the images to fit instead of resizing which causes interpolation
                size=np.shape(tempy)
                width=size[0]
                height=size[1]
                target_width=0
                target_height=0

                #Determine which dimension is greater
                if height>width:
                    while target_height<height:
                        target_height=target_height+imgHeight
                        target_width=target_height
                else:
                    while target_height<width:
                        target_width=target_width+imgWidth
                        target_height=target_width
                #Anti-Crop the image
                mask=tempMask.crop((0,0,target_height,target_width))
                mask=img_to_array(mask)

                numRows=round((target_height/imgHeight))
                numCols=round((target_width/imgWidth))
                croppedMasks=[]
                for i in range(numRows):
                    for j in range (numCols):
                        tempy=mask
                        x1=imgWidth*(i)
                        x2=imgWidth*(i+1)
                        y1=imgHeight*(j)
                        y2=imgHeight*(j+1)
                        tempy=tempy[x1:x2, y1:y2, 0:1]
                        croppedMasks.append(tempy)

            #Save images. Squeeze basically just puts an array of numbers into another, usually smaller (dimensionally) array
            #X[n,...,0]=x_img.squeeze()
            for i in range(len(croppedImgs)):
                X.append(croppedImgs[i])

            if train:
                #y[n]=mask
                for i in range(len(croppedMasks)):
                    y.append(croppedMasks[i])

        print('All images have been retrived!')
        if train:
            return X, y
        else:
            return X

    #Here we take our list of images and convert them onto numpy arrays
    def listToArray(X,y,minPercent):
        #First we will want to toss any arrays that conatain only zeros
        newX=[]
        newY=[]
        percentOnes=0
        for i in range(len(X)):
            tempX=X[i]
            tempy=y[i]
            tempPercent=(np.sum(tempy)/(imgHeight*imgWidth))
            if tempPercent>minPercent:
                percentOnes=percentOnes+(np.sum(tempy)/(imgHeight*imgWidth))
                newX.append(tempX)
                newY.append(tempy)
        #Create numpy arrays to hold the images
        X=np.zeros((len(newX), imgHeight, imgWidth, 1), dtype=np.float32)
        y=np.zeros((len(newY), imgHeight, imgWidth, 1), dtype=np.float32)
        print('Percentage of Ones in Masks: '+ str(percentOnes/len(newY)))
        for n in range(len(newX)):
            X[n,...,0]=newX[n].squeeze()
            y[n]=newY[n]
        return X,y

    #Here we create a function that preps the masks to be used in the makeWeightMap function. We need to shave the depth channel from the masks
    def weightPrep(masks):
        #Create a numpy array to hold the modified masks
        modifiedMasks=np.zeros((len(masks), imgHeight, imgWidth), dtype=np.float32)
        #Iterate through each mask and shave the channel dimension
        for i in range(len(masks)):
            tempMask=masks[i][:,:,0]
            modifiedMasks[i]=tempMask

        return modifiedMasks

    #This is the weight map generator that Matt sent me
    def unetwmap(mask, w0=15, sigma=5):
        uvals=np.unique(mask)
        wmp=np.zeros(len(uvals))
        for i, uv in enumerate(uvals):
            wmp[i]=1/np.sum(mask==uv)
        #Normalize
        wmp=wmp/wmp.max()
        wc=np.float32(mask)
        for i, uv in enumerate(uvals):
            wc[mask==uv]=wmp[i]
        #Convert to bwlabel
        cells=label(mask==1, neighbors=4)
        #Cell distance map
        bwgt=np.zeros(mask.shape)
        maps=np.zeros((mask.shape[0], mask.shape[1], cells.max()))
        if cells.max()>=2:
            for ci in range(cells.max()):
                maps[:,:,ci]=ndimage.distance_transform_edt(np.invert(cells==ci+1))
            maps=np.sort(maps,2)
            d0=maps[:,:,0]
            d1=maps[:,:,1]
            bwgt=w0*np.exp(-np.square(d0+d1)/(2*sigma))*(cells==0)
        #Unet weights
        weight=wc+bwgt
        return weight

    def unetWeightMaster(masks, w0, sigma):
        weightMaps=np.zeros((len(masks),masks.shape[1], masks.shape[1]))
        for i in range(len(masks)):
            weightMaps[i]=unetwmap(masks[i], w0, sigma)
        return weightMaps


    #Now Lets call our getData function to get our training images and their masks
    X, y=getData(pathTrain, imageType, train=True)
    print(len(X))
    print(len(y))
    X, y=listToArray(X,y, MinPercentWhite)
    print(len(X))
    print(len(y))
    #Get average value of pixel in array then divide array by that to normalize brightness
    xMean=np.mean(X)
    #Normalize
    X=X/xMean

    #Split training into training and validation
    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=ValidationSplit, random_state=2018)


    #clear the graph so all the variables are on the same graph
    tf.reset_default_graph
    #clear the keras backend
    keras.backend.clear_session()

    # Here we create a block of two convolutions with batch normalization and Relu activation
    def conv2dBlock(inputTensor, numFilters, activation, kernalSize, padding='same', batchNorm=True):
        #Here is the First convolution
        tempC1=layers.Conv2D(numFilters, (kernalSize, kernalSize), activation=activation, padding=padding)(inputTensor)

        #If batchNorm is True do this, else don't
        if batchNorm:
            #Note, BatchNormalization has many options that can be tweaked, including momentum
            tempC1=layers.BatchNormalization()(tempC1)

        #Apply the type of activation requested, by default this is relu
        #tempC=layers.Activation(activation)

        #Now we perform the second convolution of this Block. The code is almost exaclty the same as above
        #The only difference is we perform the convlution on tempC, instead of inputTensor
        tempC2=layers.Conv2D(numFilters, (kernalSize, kernalSize), activation=activation, padding=padding)(tempC1)

        #If batchNorm is True do this, else don't
        if batchNorm:
            #Note, BatchNormalization has many options that can be tweaked, including momentum
            tempC2=layers.BatchNormalization()(tempC2)

        #Apply the type of activation requested, by default this is relu
        #tempC2=layers.Activation(activation)

        #Now we return our new tensor!
        return tempC2

    #Create a model for small focal adhesion identification
    def ModelSmall(x_train, y_train, x_valid, y_valid, test=False, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 1), name='img')
        activation='relu'
        kernalSize=SmallKernal
        dropout=0.81
        #Descent down the U of UNet (Contracting)

        #Here, we begin with our first convolution block. At this point, the filters will be learning low-level features, like edges
        #We make use of the function we built above to perform two convolutions with activation (default relu) and batch normalization when requested (default True)
        #256
        c1=conv2dBlock(inputImg, numFilters*1, activation, kernalSize)

        #Now we perform MaxPooling to reduce the image size and find the most important pixels found by the convolution block
        p1=layers.MaxPooling2D((2,2))(c1)

        #Now we perform Dropout, which forces the neural network to not be able to use randomly assinged neurons, preventing the neural newtork from becoming overly
        #dependent on any single neuron, improving generalization by decreasing the risk/degree of overfitting
        p1=layers.Dropout(dropout*0.5)(p1)


        #On to the next layer of convolutions, pooling, and droupout
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #128
        c2=conv2dBlock(p1, numFilters*2, activation, kernalSize)
        p2=layers.MaxPooling2D((2,2))(c2)
        p2=layers.Dropout(dropout)(p2)

        #Next layer
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #64
        c3=conv2dBlock(p2, numFilters*4, activation, kernalSize)
        p3=layers.MaxPooling2D((2,2))(c3)
        p3=layers.Dropout(dropout)(p3)

        #Final layer of the descent, At this point the filters will be learning complex features
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #32
        c4=conv2dBlock(p3, numFilters*8, activation, kernalSize)
        p4=layers.MaxPooling2D((2,2))(c4)
        p4=layers.Dropout(dropout)(p4)

        #16
        c5=conv2dBlock(p4, numFilters*16, activation, kernalSize)
        p5=layers.MaxPooling2D((2,2))(c5)
        p5=layers.Dropout(dropout)(p5)

        ########################################################################################################################################################
        #Now we have hit the bottleneck portion of the UNet architecture. Here we perform only one convolution block and no MaxPooling or Droupout
        #8
        c6=conv2dBlock(p5, numFilters*32, activation, kernalSize)

        ########################################################################################################################################################
        #Now we begin ascending the U (expansion).

        #First we perform an up convolution (also called an Transposed Convolution) on the bottleNeck
        #8
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize, kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize, kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize, kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize, kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize, kernalSize), strides=(2,2), padding='same')(c10)
        u10=layers.concatenate([u10, c1], axis=3)
            #Note, I am unsure of why the axis=3 part here
        u10=layers.Dropout(dropout)(u10)
        c11=conv2dBlock(u10, numFilters*1, activation, kernalSize)


        #256

        #Now we perform on last 'convolution' that compreses the depth of the image to 1. That is, take all the images created by the filters and combine them into one image.
        outputs=layers.Conv2D(2,(1,1), activation='sigmoid',name="outputs")(c11)


        #Create the actual model
        modelSmall= models.Model(inputs=inputImg ,outputs=outputs)
        opt = keras.optimizers.Adagrad()


        #Compile model
        modelSmall.compile(optimizer=opt, loss='logcosh', metrics=["accuracy"])
        if test==False:
            #Create callbacks
            callbacksSmall = [
            keras.callbacks.ModelCheckpoint(modelSmallSaveName, verbose=1, save_best_only=True, save_weights_only=True)
            ]

            #Train the first model
            resultsSmall = modelSmall.fit(x_train, y_train, batch_size=SmallBatch, epochs=SmallEpochs, callbacks=callbacksSmall, validation_data=(x_valid, y_valid), class_weight=[1, 1])

            return modelSmall
        else:
            #load in the best model which we have saved
            modelSmall.load_weights(modelSmallSaveName)
            # Predict on train, val and test
            preds_train = modelSmall.predict(x_train, verbose=1)
            preds_val = modelSmall.predict(x_valid, verbose=1)
            smallX=preds_train
            smallValid=preds_val
            return smallX, smallValid


    #Create a model for seperating boundary focal adhesion identification
    def ModelBoundary(x_train, y_train, x_valid, y_valid, test=False, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 1), name='img')
        activation='relu'
        dropout=0.72
        kernalSize=BoundaryKernal
        _epsilon = tf.convert_to_tensor(keras.backend.epsilon(), np.float32)
        #Here we create a new input that will be for the weight maps

        weightMapInput=layers.Input(shape=(imgHeight, imgWidth,1))

        #Make second set of training and validation Masks for Boundary-focused UNet
        #Create Training Masks
        tempMasks=weightPrep(y_train)
        #Create the weight maps
        weightMaps=unetWeightMaster(tempMasks, w0, sigma)
        #Reshape to add back in channel dimension so that it is compatible with the model
        weightMaps=resize(weightMaps,(len(weightMaps),imgHeight,imgWidth,1), mode='constant', preserve_range=True, anti_aliasing =True, order=5)
        #Create Validation Masks
        tempMasks=weightPrep(y_valid)
        #Create the weight maps
        weightMapsValid=unetWeightMaster(tempMasks, w0, sigma)
        weightMapsValid=resize(weightMapsValid,(len(weightMapsValid),imgHeight,imgWidth,1), mode='constant', preserve_range=True, anti_aliasing =True, order=5)

        #Descent down the U of UNet (Contracting)

        #Here, we begin with our first convolution block. At this point, the filters will be learning low-level features, like edges
        #We make use of the function we built above to perform two convolutions with activation (default relu) and batch normalization when requested (default True)
        #256
        c1=conv2dBlock(inputImg, numFilters*1, activation, kernalSize)

        #Now we perform MaxPooling to reduce the image size and find the most important pixels found by the convolution block
        p1=layers.MaxPooling2D((2,2))(c1)

        #Now we perform Dropout, which forces the neural network to not be able to use randomly assinged neurons, preventing the neural newtork from becoming overly
        #dependent on any single neuron, improving generalization by decreasing the risk/degree of overfitting
        p1=layers.Dropout(dropout*0.5)(p1)


        #On to the next layer of convolutions, pooling, and droupout
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #128
        c2=conv2dBlock(p1, numFilters*2, activation, kernalSize)
        p2=layers.MaxPooling2D((2,2))(c2)
        p2=layers.Dropout(dropout)(p2)

        #Next layer
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #64
        c3=conv2dBlock(p2, numFilters*4, activation, kernalSize)
        p3=layers.MaxPooling2D((2,2))(c3)
        p3=layers.Dropout(dropout)(p3)

        #Final layer of the descent, At this point the filters will be learning complex features
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #32
        c4=conv2dBlock(p3, numFilters*8, activation, kernalSize)
        p4=layers.MaxPooling2D((2,2))(c4)
        p4=layers.Dropout(dropout)(p4)

        #16
        c5=conv2dBlock(p4, numFilters*16, activation, kernalSize)
        p5=layers.MaxPooling2D((2,2))(c5)
        p5=layers.Dropout(dropout)(p5)

        ########################################################################################################################################################
        #Now we have hit the bottleneck portion of the UNet architecture. Here we perform only one convolution block and no MaxPooling or Droupout
        #8
        c6=conv2dBlock(p5, numFilters*32, activation, kernalSize)

        ########################################################################################################################################################
        #Now we begin ascending the U (expansion).

        #First we perform an up convolution (also called an Transposed Convolution) on the bottleNeck
        #8
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize, kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize, kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize, kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize, kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize, kernalSize), strides=(2,2), padding='same')(c10)
        u10=layers.concatenate([u10, c1], axis=3)
            #Note, I am unsure of why the axis=3 part here
        u10=layers.Dropout(dropout)(u10)
        c11=conv2dBlock(u10, numFilters*1, activation, kernalSize)


        #256

        #Now we perform on last 'convolution' that compreses the depth of the image into the number of classes we have
        softmaxOp=layers.Conv2D(2,(1,1), activation='softmax', name='softmaxOp')(c11)

        #Instead of computing Loss outside of the model, we are going to intergrate it into the model itself
        #These layers are non-trainable and mimic the computation of logcosh loss
        #The actual loss function will onlly need to perform the aggregation
        logcosh=layers.Lambda(lambda x: x + keras.backend.softplus(-2.*x)- keras.backend.log(2.))(softmaxOp)


        weightedSoftmax=keras.layers.multiply([logcosh,weightMapInput])

        #Create the actual model
        modelBoundary= models.Model(inputs=[inputImg, weightMapInput], outputs=[weightedSoftmax])
        #set optimizer
        opt = keras.optimizers.Adagrad()

        #Create custom loss
        def customLoss(y_true, y_pred):
            return -tf.reduce_sum(y_true*y_pred, len(y_pred.get_shape())-1)
        #Complile models
        modelBoundary.compile(optimizer=opt, loss=customLoss, metrics=["accuracy"])
        if test==False:
            callbacksBoundary = [
                keras.callbacks.ModelCheckpoint(modelBoundarySaveName, verbose=1, save_best_only=True, save_weights_only=True)
                ]
             #Train small Model
            resultBoundary = modelBoundary.fit([x_train, weightMaps], y_train, batch_size=BoundaryBatch, epochs=BoundaryEpochs, callbacks=callbacksBoundary, validation_data=([x_valid, weightMapsValid], y_valid))
            return modelBoundary
        else:
            #load in the best model which we have saved
            modelBoundary.load_weights(modelBoundarySaveName)
            #Create a new model using those weights that no longer uses weightMaps, i.e is noraml UNet
            layerName='softmaxOp'
            newModel=models.Model(inputs=modelBoundary.input[0], outputs=modelBoundary.get_layer(layerName).output)
            #complile new model
            newModel.compile(optimizer=keras.optimizers.Adagrad(), loss=customLoss, metrics=["accuracy"])
            #Save NewModel weights
            newModel.save_weights(NewBoundarySaveName)
            # Predict on train, val and test
            preds_train = newModel.predict(x_train, verbose=1)
            preds_val = newModel.predict(x_valid, verbose=1)
            boundaryX=preds_train
            boundaryValid=preds_val
            return boundaryX, boundaryValid

    #Create a model to analyze combined masks
    def ModelMixer(newX, y_train, newValidX, y_val, test=False, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 5), name='img')
        activation='elu'
        dropout=0.72
        kernalSize=MixerKernal
        #Descent down the U of UNet (Contracting)

        #Here, we begin with our first convolution block. At this point, the filters will be learning low-level features, like edges
        #We make use of the function we built above to perform two convolutions with activation (default relu) and batch normalization when requested (default True)
        #256
        c1=conv2dBlock(inputImg, numFilters*1, activation, kernalSize)

        #Now we perform MaxPooling to reduce the image size and find the most important pixels found by the convolution block
        p1=layers.MaxPooling2D((2,2))(c1)

        #Now we perform Dropout, which forces the neural network to not be able to use randomly assinged neurons, preventing the neural newtork from becoming overly
        #dependent on any single neuron, improving generalization by decreasing the risk/degree of overfitting
        p1=layers.Dropout(dropout*0.5)(p1)


        #On to the next layer of convolutions, pooling, and droupout
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #128
        c2=conv2dBlock(p1, numFilters*2, activation, kernalSize)
        p2=layers.MaxPooling2D((2,2))(c2)
        p2=layers.Dropout(dropout)(p2)

        #Next layer
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #64
        c3=conv2dBlock(p2, numFilters*4, activation, kernalSize)
        p3=layers.MaxPooling2D((2,2))(c3)
        p3=layers.Dropout(dropout)(p3)

        #Final layer of the descent, At this point the filters will be learning complex features
        #We begin by calling the convolution block on the results from the last layer's dropout. We also double the number of filters
        #32
        c4=conv2dBlock(p3, numFilters*8, activation, kernalSize)
        p4=layers.MaxPooling2D((2,2))(c4)
        p4=layers.Dropout(dropout)(p4)

        #16
        c5=conv2dBlock(p4, numFilters*16, activation, kernalSize)
        p5=layers.MaxPooling2D((2,2))(c5)
        p5=layers.Dropout(dropout)(p5)

        ########################################################################################################################################################
        #Now we have hit the bottleneck portion of the UNet architecture. Here we perform only one convolution block and no MaxPooling or Droupout
        #8
        c6=conv2dBlock(p5, numFilters*32, activation, kernalSize)

        ########################################################################################################################################################
        #Now we begin ascending the U (expansion).

        #First we perform an up convolution (also called an Transposed Convolution) on the bottleNeck
        #8
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize, kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize, kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize, kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize, kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize, kernalSize), strides=(2,2), padding='same')(c10)
        u10=layers.concatenate([u10, c1], axis=3)
            #Note, I am unsure of why the axis=3 part here
        u10=layers.Dropout(dropout)(u10)
        c11=conv2dBlock(u10, numFilters*1, activation, kernalSize)


        #256

        #Now we perform on last 'convolution' that compreses the depth of the image to 1. That is, take all the images created by the filters and combine them into one image.
        outputs=layers.Conv2D(2,(1,1), activation='sigmoid', name="outputs")(c11)

        #Create the actual model
        modelMixer= models.Model(inputs=inputImg ,outputs=outputs)

        #set optimizer
        opt = keras.optimizers.Adagrad()

        #Complile models
        modelMixer.compile(optimizer=opt, loss='logcosh', metrics=["accuracy"])
        if test==False:
            callbacksMixer= [
                    keras.callbacks.ModelCheckpoint(modelMixerSaveName, verbose=1, save_best_only=True, save_weights_only=True)
                ]
            #Train small Model
            out = modelMixer.fit(newX, y_train, batch_size=MixerBatch, epochs=MixerEpochs, validation_data=(newValidX, y_val), callbacks=callbacksMixer)
            return out, modelMixer
        else:
            #load in the best model which we have saved
            modelMixer.load_weights(modelMixerSaveName)
            # Predict on train, val and test
            preds_train = modelMixer.predict(newX, verbose=1)
            preds_val = modelMixer.predict(newValidX, verbose=1)
            return preds_train, preds_val


    def TrainModel(x_train, y_train, x_val, y_val):
        #Process model results to get ready for next model
        modelSmall=ModelSmall(x_train, y_train, x_val, y_val)
        print('Finished Small Model')
        #load in the best model which we have saved
        modelSmall.load_weights(modelSmallSaveName)
        # Predict on train, val
        preds_train = modelSmall.predict(x_train, verbose=1)
        preds_val = modelSmall.predict(x_val, verbose=1)
        #Set new values
        smallX=preds_train
        smallValidX=preds_val
        #Thresh
        if True==True:
            thresh=SmallThresh
            smallX=1*(smallX>thresh)
            smallValidX=1*(smallValidX>thresh)
        #clear the graph so all the variables are on the same graph
        #tf.reset_default_graph
        #clear the keras backend
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************


        #Process model results to get ready for next model
        modelBoundary=ModelBoundary(x_train, y_train, x_val, y_val)
        print('Finished Boundary Model')
        #load in the best model which we have saved
        modelBoundary.load_weights(modelBoundarySaveName)
        #Create a new model using those weights that no longer uses weightMaps, i.e is noraml UNet
        layerName='softmaxOp'
        newModelBoundary=models.Model(inputs=modelBoundary.input[0], outputs=modelBoundary.get_layer(layerName).output)
        #complile new model
        newModelBoundary.compile(optimizer=keras.optimizers.Adagrad(), loss='logcosh', metrics=["accuracy"])
        # Predict on train, val and test
        preds_train = newModelBoundary.predict(x_train, verbose=1)
        preds_val = newModelBoundary.predict(x_val, verbose=1)
        boundaryX=preds_train
        boundaryValidX=preds_val
        #Thresh
        if True==True:
            thresh=BoundaryThresh
            boundaryX=1*(boundaryX>thresh)
            boundaryValidX=1*(boundaryValidX>thresh)
        #Combine Masks
        newMasks=np.concatenate((smallX,boundaryX ),-1)
        newMasksValid=np.concatenate((smallValidX,boundaryValidX ),-1)
        #Combine new Masks with orginal images
        newX=np.concatenate((newMasks, x_train ),-1)
        newValidX=np.concatenate((newMasksValid, x_val),-1)
        print(np.shape(newX))
        #normalize
        newX=newX/np.mean(newX)
        newValidX=newValidX/np.mean(newValidX)

        #clear the graph so all the variables are on the same graph
        #tf.reset_default_graph
        #clear the keras backend
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************

        out, model=ModelMixer(newX, y_train, newValidX, y_val)
        return out, model


    def TestModel(x_train, y_train, x_val, y_val):
        #Process model results to get ready for next model
        smallX, smallValidX=ModelSmall(x_train, y_train, x_val, y_val, test=True)
        #Thresh
        if True==True:
            thresh=SmallThresh
            smallX=1*(smallX>thresh)
            smallValidX=1*(smallValidX>thresh)
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************

        #Process model results to get ready for next model
        boundaryX, boundaryValidX=ModelBoundary(x_train, y_train, x_val, y_val, test=True)
        print('Finished Boundary Model')
        #Thresh
        if True==True:
            thresh=BoundaryThresh
            boundaryX=1*(boundaryX>thresh)
            boundaryValidX=1*(boundaryValidX>thresh)

        #Combine Masks
        newMasks=np.concatenate((smallX,boundaryX ),-1)
        newMasksValid=np.concatenate((smallValidX,boundaryValidX ),-1)
        #Combine new Masks with orginal images
        newX=np.concatenate((newMasks, x_train ),-1)
        newValidX=np.concatenate((newMasksValid, x_val),-1)
        #normalize
        newX=newX/np.mean(newX)
        newValidX=newValidX/np.mean(newValidX)
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************

        pred1, pred2=ModelMixer(newX, y_train, newValidX, y_val, test=True)
        return pred1, pred2


    def MasterModel(x_train, y_train, x_val, y_val, test=False):
        keras.backend.clear_session()
        if test==False:
            out, model=TrainModel(x_train, y_train, x_val, y_val)
            return out, model
        else:
            pred1, pred2=TestModel(x_train, y_train, x_val, y_val)
            return pred1, pred2

    #Run model
    results, model=MasterModel(XTrain, yTrain, XValid, yValid)
    #Test Once To Get newBoundary weights
    preds_train, preds_val = MasterModel(XTrain, yTrain, XValid, yValid, test=True)
    print('End Training')