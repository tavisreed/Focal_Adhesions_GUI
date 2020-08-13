#!/usr/bin/env python
# coding: utf-8

# ***
# This Version uses the model from UNet2_FA_V7
# ***
# 
# This is for processing Focal Adhesion Images

#Here we import all the libraries that we will need
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
#This can be used to help find a pattern
import fnmatch
#This is for preprocessing the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#These are also for image preprocessing
from skimage.transform import resize
#For the sigmoid function: math.exp(-x)
import math
from PIL import Image
#To read and display images
import cv2
#For watershedding
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max


def segment(imageReplaceType, saveType, ImageFolder,
            MaskFolder, ModelName, thresh):
    print('Start Segmenting')

    # Some important parameters that will be used later
    # Image Related
    imgWidth = 64
    imgHeight = 64
    numChannles = 1
    imageReplaceType = '.' + imageReplaceType.replace('.', '')
    imageType = '*' + imageReplaceType
    saveType = '.' + saveType
    pathTrain = '..\\GUI\\Raw_Images\\' + ImageFolder + '\\'
    #Check to see if Mask folder exists. If not, create it
    if not os.path.exists('..\\GUI\\Masks\\' + MaskFolder):
        os.makedirs('..\\GUI\\Masks\\' + MaskFolder)
    maskSavePath = '..\\GUI\\Masks\\' + MaskFolder + '\\'
    # model save name
    modelSmallSaveName = '..\\GUI\\Segment\\Saved_Models\\' + ModelName + '\\small.h5'
    modelBoundarySaveName = '..\\GUI\\Segment\\Saved_Models\\' + ModelName + '\\boundary.h5'
    NewBoundarySaveName = '..\\GUI\\Segment\\Saved_Models\\' + ModelName + '\\newboundary.h5'
    modelMixerSaveName = '..\\GUI\\Segment\\Saved_Models\\' + ModelName + '\\mixer.h5'
    thresh = thresh



    #Here we have a function that just retrives relevent file names
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(name)
        return result

    #Here we create a function for getting data that can be used to retreive training images, their masks, and test images
    def getData(path, imageType, imageReplaceType, imgHeight, imgWidth):
        #Here we are getting the id of all the images stored in the folder
        ids=find(imageType, path)
        for i in range(len(ids)):
            ids[i] = int(ids[i].replace(imageReplaceType, ''))
        ids.sort()
        for i in range(len(ids)):
            ids[i] = str(ids[i])

        #We create an array that will hold the images. dtype=np.float32 sets the data type of the array to a 32 bit float
        X=[]
        orderX=[]

        #Here we are actually getting each image and putting them in their respective array. We use tqdm_notebook to create a loading bar
        for n, tempId in tqdm_notebook(enumerate(ids),total=len(ids)):
            #Before we can load the image, we need to do some quick edits to the ID string to make it easier to get our images
            tempId=tempId.replace(imageReplaceType, '')
            #Load in the images
            tempImage=cv2.imread(path+tempId+imageReplaceType)
            tempImage=array_to_img(tempImage[:,:,0:1])
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
            numImgs=0
            for i in range(numRows):
                for j in range (numCols):
                    numImgs=numImgs+1
                    tempy=x_img
                    x1=imgWidth*(i)
                    x2=imgWidth*(i+1)
                    y1=imgHeight*(j)
                    y2=imgHeight*(j+1)
                    tempy=tempy[x1:x2, y1:y2, :]
                    croppedImgs.append(tempy)

            #Save images. Squeeze basically just puts an array of numbers into another, usually smaller (dimensionally) array
            orderX.append(numImgs)
            for i in range(len(croppedImgs)):
                X.append(croppedImgs[i])
        return X, orderX

    #Here we create a function for getting the true images without croppings
    def getTrueImages(path, imageType, imageReplaceType, imgHeight, imgWidth):
        #Here we are getting the id of all the images stored in the folder
        ids=find(imageType, path)
        for i in range(len(ids)):
            ids[i] = int(ids[i].replace(imageReplaceType, ''))
        ids.sort()
        for i in range(len(ids)):
            ids[i] = str(ids[i])
        #We create an array that will hold the images. dtype=np.float32 sets the data type of the array to a 32 bit float
        X=[]
        #Here we are actually getting each image and putting them in their respective array. We use tqdm_notebook to create a loading bar
        for n, tempId in tqdm_notebook(enumerate(ids),total=len(ids)):
            #Before we can load the image, we need to do some quick edits to the ID string to make it easier to get our images
            tempId=tempId.replace(imageReplaceType, '')
            #Load in the images
            tempImage=cv2.imread(path+tempId+imageReplaceType)
            tempImage=array_to_img(tempImage[:,:,0:1])
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
            X.append(x_img)
        return X

    #Here we take our list of images and convert them onto numpy arrays
    def listToArray(X, imgHeight, imgWidth):
        #First we will want to toss any arrays that conatain only zeros
        newX=[]
        for i in range(len(X)):
            tempX=X[i]
            newX.append(tempX)
        #Create numpy arrays to hold the images
        X=np.zeros((len(newX), imgHeight, imgWidth, 1), dtype=np.float32)
        for n in range(len(newX)):
            X[n,...,0]=newX[n].squeeze()
        return X

    def ImageReconstructor(X, orderX, imgHeight, imgWidth):
        placeX=0
        listX=[]
        newX=[]
        #First lets gather images that were originally one into a list and then put that into a list
        for i in range(len(orderX)):
            tempListX=[]
            for j in range(orderX[i]):
                idx=placeX+j
                img=array_to_img(X[idx]*255)
                tempListX.append(img)
            placeX=placeX+orderX[i]
            listX.append(tempListX)
        #Now lets actually reconstruct the image
        for i in range(len(listX)):
            #Get image width and height
            currentSet=listX[i]
            numRows=int(math.sqrt(len(currentSet)))
            numCols=numRows
            width=numRows*imgWidth
            height=numCols*imgHeight
            #Make a new blank image
            img=Image.new('RGB', (width, height))
            #paste image parts into new image
            idx=0
            for i in range(numRows):
                for j in range (numCols):
                    x1=imgWidth*(i)
                    y1=imgHeight*(j)
                    img.paste(currentSet[idx], box=(y1,x1))
                    idx=idx+1
            newX.append(img)
        return newX

    #Here we create a function so save images
    def saver(masks, saveType, maskSavePath):
        print("Saving!!")
        for i in tqdm_notebook(range(len(masks))):
            #Note: Need to multiple images by 255 to resotre their original range in order for them to save properly
            cv2.imwrite(maskSavePath+str(i)+saveType, img_to_array(masks[i]))
        return


    #Now Lets call our getData function to get our training images and their masks
    X, order=getData(pathTrain, imageType, imageReplaceType, imgHeight, imgWidth)
    #Create function to get the origianl images, those without being chooped up
    trueX=getTrueImages(pathTrain, imageType, imageReplaceType, imgHeight, imgWidth)
    X=listToArray(X, imgHeight, imgWidth)
    xMean=np.mean(X)
    #Normalize
    X=X/xMean


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
    def ModelSmall(X, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 1), name='img')
        activation='relu'
        kernalSize=3
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
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize,kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize,kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize,kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize,kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize,kernalSize), strides=(2,2), padding='same')(c10)
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
        #load in the best model which we have saved
        modelSmall.load_weights(modelSmallSaveName)
        # Predict on train, val and test
        preds= modelSmall.predict(X, verbose=1)
        smallX=preds
        return smallX


    #Create a model for seperating boundary focal adhesion identification
    def ModelBoundary(X, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 1), name='img')
        activation='relu'
        dropout=0.72
        kernalSize=5
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
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize,kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize,kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize,kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize,kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize,kernalSize), strides=(2,2), padding='same')(c10)
        u10=layers.concatenate([u10, c1], axis=3)
            #Note, I am unsure of why the axis=3 part here
        u10=layers.Dropout(dropout)(u10)
        c11=conv2dBlock(u10, numFilters*1, activation, kernalSize)


        #256

        #Now we perform on last 'convolution' that compreses the depth of the image into the number of classes we have
        softmaxOp=layers.Conv2D(2,(1,1), activation='softmax', name='softmaxOp')(c11)

        #Create the actual model
        modelBoundary= models.Model(inputs=[inputImg], outputs=[softmaxOp])

        #Create custom loss
        def customLoss(y_true, y_pred):
            return -tf.reduce_sum(y_true*y_pred, len(y_pred.get_shape())-1)
        #complile new model
        modelBoundary.compile(optimizer=keras.optimizers.Adagrad(), loss=customLoss, metrics=["accuracy"])
        #load in the best model which we have saved
        modelBoundary.load_weights(NewBoundarySaveName)
        # Predict on train, val and test
        preds = modelBoundary.predict(X, verbose=1)
        boundaryX=preds
        return boundaryX


    #Create a model to analyze combined masks
    def ModelMixer(newX, numFilters=16, numClass=1):
        inputImg = layers.Input(shape=(imgHeight, imgWidth, 5), name='img')
        activation='elu'
        dropout=0.72
        kernalSize=9
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
        u6=layers.Conv2DTranspose(numFilters*16,(kernalSize,kernalSize), strides=(2,2), padding='same')(c6)

        #Next we concatenate this Transposed convolution with the convolution of the corresponding size that accured during the descent
        u6=layers.concatenate([u6, c5])
            #Note, I have also seen upConcat1= layers.concatenate([upConv1, conv4], axis=concat_axis), where concat_axis=3, but I am unsure why this was used

        #We now perform Dropout on the concatenation
        u6=layers.Dropout(dropout)(u6)

        #We now perform a convolution block
        c7=conv2dBlock(u6, numFilters*16, activation, kernalSize)

        #Now we move on to the next layer of the expansion. Here we halve the number of filters
        #16
        u7=layers.Conv2DTranspose(numFilters*8,(kernalSize,kernalSize), strides=(2,2), padding='same')(c7)
        u7=layers.concatenate([u7, c4])
        u7=layers.Dropout(dropout)(u7)
        c8=conv2dBlock(u7, numFilters*8, activation, kernalSize)

        #Next layer
        #32
        u8=layers.Conv2DTranspose(numFilters*4,(kernalSize,kernalSize), strides=(2,2), padding='same')(c8)
        u8=layers.concatenate([u8, c3])
        u8=layers.Dropout(dropout)(u8)
        c9=conv2dBlock(u8, numFilters*4, activation, kernalSize)

        #Final layer of the expansion
        #64
        u9=layers.Conv2DTranspose(numFilters*2,(kernalSize,kernalSize), strides=(2,2), padding='same')(c9)
        u9=layers.concatenate([u9, c2])
        u9=layers.Dropout(dropout)(u9)
        c10=conv2dBlock(u9, numFilters*2, activation, kernalSize)


        #128
        u10=layers.Conv2DTranspose(numFilters*1,(kernalSize,kernalSize), strides=(2,2), padding='same')(c10)
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
        #load in the best model which we have saved
        modelMixer.load_weights(modelMixerSaveName)
        # Predict on train, val and test
        preds= modelMixer.predict(newX, verbose=1)
        return preds


    def TestModel(X):
        #Process model results to get ready for next model
        smallX=ModelSmall(X)
        print('Finished Small Model')
        #Thresh
        thresh=0.8
        smallX=1*(smallX>thresh)
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************
        #Process model results to get ready for next model
        boundaryX=ModelBoundary(X)
        print('Finished Boundary Model')
        #Thresh
        thresh=0.999999977
        boundaryX=1*(boundaryX>thresh)
        #Combine Masks
        newMasks=np.concatenate((smallX,boundaryX ),-1)
        #Combine new Masks with orginal images
        newX=np.concatenate((newMasks, X),-1)
        #normalize
        newX=newX/np.mean(newX)
        keras.backend.clear_session()
        #***************************************************
        #***************************************************
        #***************************************************

        preds=ModelMixer(newX)
        return preds


    def MasterModel(X):
        keras.backend.clear_session()
        preds=TestModel(X)
        return preds


    #Generate Masks
    preds=MasterModel(X)

    preds = np.maximum(preds[:,:,:,0],preds[:,:,:,1])

    #Restructure Masks to add in 3rd dimeison so that it works with the ImageReconstructor function
    preds=resize(preds,(len(preds),imgHeight,imgWidth,1), mode='constant', preserve_range=True, anti_aliasing =True, order=5)



    preds_threshed=1*(preds>=thresh)

    #Reconstruct the masks
    masks=ImageReconstructor(preds_threshed, order, imgHeight, imgWidth)

    saver(masks, saveType, maskSavePath)
    print('Segmenting Finished')



