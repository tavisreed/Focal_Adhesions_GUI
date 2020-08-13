#!/usr/bin/env python
# coding: utf-8

# ***
# Using centroid tracking object tracking algorithm. This method relies on the Euclidean distance between existing object centroids and new object centroids between subsequent frames in a video
# ***
# 
# Created following following tutorials: 
# - https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# - https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/


from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import os
from tqdm import tqdm_notebook
import fnmatch
import csv




#Here we create a class for tracking the centroids
class CentroidTracker:
    #The intialization function
    def __init__(self, maxDisappeared=1):
        #In this function we are initalizing the next unique object ID
        #along with two orderd dictionaries that are used to keep track
        #of mapping a given object ID to its centroid and the number of
        #consecutaive frames that it has been marked "disappeared"
        
        #This is a counter used to assign unique IDs to each object
        self.nextObjectID=0
        #This is a dictonary that uses the object ID as the key and
        #the centroid (x,y) coordinates as the value
        self.centroids=OrderedDict()
        #This is a dictonary that uses the object ID as the key and the
        #box x1,y1,x2,y2 coordiantes as the value
        self.pixels=OrderedDict()
        
        #This is a dictonary that uses the object ID as the key, and
        #the number of consecutive frames an object has been marked lost
        self.disappeared=OrderedDict()
        
        #Here we are storing the maximum consecutive frames a given object
        #is allowed to be marked as disappeared until we deregister the
        #object from tracking
        self.maxDisappeared=maxDisappeared
    
    #Function to register a new object to our tracker
    def register(self, centroid, pixels):
        #When registering an object, we use the next avaliable object
        #ID to store the centroid
        self.centroids[self.nextObjectID]=centroid
        self.pixels[self.nextObjectID]=pixels
        self.disappeared[self.nextObjectID]=0
        self.nextObjectID+=1
        
    #Function to deregister from out tracker an old object 
    #that has been lost
    def deregister(self, objectID):
        #to deregister an object ID we delete the object ID from
        #both of our respective dictionaries
        del self.centroids[objectID]
        del self.pixels[objectID]
        del self.disappeared[objectID]
    
    #Function that updates the location of objects by taking in a list
    #of bounding box rectangles
    def update(self, rects, FAPixels):
        #Assumed structure of rects parameter is a tuple with
        #the structure (startX, startY, endX, endY)
        
        #Check to see if the list of input bounding box rectangles is
        #empty
        if len(rects)==0:
            #loop over all existing tracked objects and mark 
            #them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID]+=1
                
                #If we have reached a maximum number of consecutive frames
                #for an object to be lost, deregister it
                if self.disappeared[objectID]>self.maxDisappeared:
                    self.deregister(objectID)
            #Return early since there are no centroids or tracking
            #info to update
            return self.centroids, self.pixels
        
        #Initialize an array of input centroids for the current frame
        inputCentroids=np.zeros((len(rects),2), dtype="int")
        inputPixels=[]

        #loop over the bounding box rectangles
        for (i,(x1, y1, x2, y2)) in enumerate(rects):
            #use the bounding box coordinates to derive the centroid
            cX=int((x1+x2)/2.0)
            cY=int((y1+y2)/2.0)
            inputCentroids[i]=(cX,cY)
            inputPixels.append(FAPixels[i])
            
        #If we are currently not tracking any objects, take the input
        #centroids and register each of them
        if len(self.centroids)==0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputPixels[i])
        #Otherwise, we are currently tracking objects so we need to
        #try to match the input centroids to existing object centroids
        else:
            #Grab the set of object IDs and corresponding centroids
            objectIDs=list(self.centroids.keys())
            objectCentroids=list(self.centroids.values())
            
            #Compute the distance between each pair of object centroids
            #and input centroids, respectively -- our goal will be to 
            #match an input centroid to an existing object centroid
            D=dist.cdist(np.array(objectCentroids), inputCentroids)
            
            #In order to perform this matching we must (1) find the
            #smallest value in each row and then (2) sort the row indexes
            #based on their minimum values so that the row with the
            #smallest value is at the front of the index list
            rows=D.min(axis=1).argsort()
            
            #Next, we perform a similar process on the columns by finding
            #the smallest value in each column and then sorting using
            #the previously computed row index list
            cols=D.argmin(axis=1)[rows]
                
            #In order to determine if we need to update, register, or
            #deregister an object we need to keep track of which of the
            #rows and column indexes we have already examined
            usedRows=set()
            usedCols=set()
            
            #Loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                #if we have already examined either the row or
                #column value before, ignore it 
                if row in usedRows or col in usedCols:
                    continue
                    
                #Otherwise, grab the object ID for the current row,
                #set its new centroid, and reset the disappeared counter
                objectID=objectIDs[row]
                self.centroids[objectID]=(inputCentroids[col])
                self.pixels[objectID]=(inputPixels[col])
                self.disappeared[objectID]=0
                
                #indicate that we have examined each of the row and column
                #indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                
            #Compute both the row and column index we have NOT yet 
            #examined
            unusedRows=set(range(0,D.shape[0])).difference(usedRows)
            unusedCols=set(range(0,D.shape[1])).difference(usedCols)
            

            #Register each new input centroid as a trackable object
            for col in unusedCols:
                self.register(inputCentroids[col], inputPixels[col])
                
            #Check and see if some of these objects have potentially 
            # disappeared by looping over the unused row indexes
            for row in unusedRows:
                #Grab the object ID for the corresponding row index
                #and increment the disappeared counter
                objectID=objectIDs[row]
                self.disappeared[objectID]+=1

                #Check to see if the number of consecutive frames the
                #object has been marked "disappearaed"
                #for wattants deregistering the object
                if self.disappeared[objectID]>self.maxDisappeared:
                    self.deregister(objectID)
        return self.centroids, self.pixels


#Here we create a class for Focal Adhesions
class FocalAdhesion():
    def __init__(self, ID, lifetime, maxIntensity, 
                     minIntensity,  InterIntList, 
                     AvgIntList, SizeList, 
                     centroid, color, pixelIndex,
                     frameNumber,mergeList, splitList):
        self.id=ID
        self.lifetime=lifetime
        self.minInt=minIntensity
        self.maxInt=maxIntensity
        self.interInt=InterIntList
        self.avgInt=AvgIntList
        self.size=SizeList
        self.centroid=centroid
        self.color=color
        self.pixelIndex=pixelIndex
        self.frameNumber=frameNumber
        self.merge=mergeList
        self.split=splitList
        self.errorFlag=False
    

    def printLists(self):
        print("Size in Each Frame:")
        print(self.size)
        print("Min Intensity in Each Frame:")
        print(self.minInt)
        print("Max Intensity in Each Frame:")
        print(self.maxInt)
        print("Intergreated Intensity in Each Frame:")
        print(self.interInt)
        print("Average Intensity in Each Frame:")
        print(self.avgInt)
        print("Center in Each Frame:")
        print(self.centroid)
        print("FA Pixel Locations in Each Frame:")
        print(self.pixelIndex)
        print("FA Color:")
        print(self.color)
        print("Frame Numbers:")
        print(self.frameNumber)


#Here we have a function that just retrives relevent file names
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(name)
    return result


#Get images
def getImages(path, imageType, imageReplaceType, replace='', contourSetting=False):
    #Here we are getting the id of all the images stored in the folder
    ids=find(imageType, path)
    if replace!='':
        for i in range(len(ids)):
            ids[i]=ids[i].replace(replace,'')
    for i in range(len(ids)):
            ids[i]=ids[i].replace(imageReplaceType,'')
    #We create an array that will hold the images.
    images=[]
    #Get Rid of duplicates
    id_set=set(ids)
    ids=list(id_set)
    #Convert IDS to ints
    for i in range(len(ids)):
        ids[i]=int(ids[i])
    ids.sort()
    #Convert Ids to strings
    for i in range(len(ids)):
        ids[i]=str(ids[i])

    #Here we are actually getting each image and putting them in their respective array. We use tqdm_notebook to create a loading bar
    for n, tempId in tqdm_notebook(enumerate(ids),total=len(ids)):
        #Before we can load the image, we need to do some quick edits to the ID string to make it easier to get our images
        tempId=tempId
        if contourSetting==True:
            #Load in the images
            
            tempImage=cv2.imread(path+tempId+imageReplaceType, cv2.CV_8UC1)
        else:
            #Load in the images
            tempImage=cv2.imread(path+replace+tempId+imageReplaceType)
        #Save image in list
        images.append(tempImage)
    return images

def detectObjects(masksContour):
    #Find the contours in the image and initialize the shape detector
    image=masksContour.copy()
    img2, cnts ,hir=cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, 
                         cv2.CHAIN_APPROX_SIMPLE)

    #Create a list to hold rectangles
    rects=[]
    FAPixels=[]
    #Loop over the contours
    for i in range(len(cnts)):
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(masksContour)
        cv2.drawContours(cimg, cnts, i, color=255, thickness=-1)
        #Get the indexes in the image where it the filled in contour is
        pts = np.where(cimg == 255)
        FAPixels.append(pts)
        #Get bounding box
        (x,y,w,h)=cv2.boundingRect(cnts[i])
        box=(x,y,x+w,y+h)
        rects.append(box)
        
    return rects, FAPixels

#Here we create a function so save images
def saver(masks, saveType, maskSavePath):
    for i in (range(len(masks))):
        #Note: Need to multiple images by 255 to resotre their original range in order for them to save properly
        cv2.imwrite(maskSavePath+str(i)+saveType, masks[i])
    return

#Here we create a function to take the intersection of two lists
def intersection(lst1, lst2):   
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3


def joinFAs(FA, listFA, video):
    # Now lets get the information for our base FA
    frameList = FA.frameNumber
    lifetime = FA.lifetime
    sizeList = FA.size
    ID = FA.id
    color = FA.color
    averageIntensityList = FA.avgInt
    intergratedIntensityList = FA.interInt
    maxIntensityList = FA.maxInt
    minIntensityList = FA.minInt
    centroidList = FA.centroid
    pixelIndexList = FA.pixelIndex
    mergeList = FA.merge
    splitList = FA.split

    # Get the current FA's frame list
    currentFrameList = listFA.frameNumber
    # Create some holder lists
    holderFrame = []
    holderSize = []
    holderAvgInt = []
    holderIntInt = []
    holderMaxInt = []
    holderMinInt = []
    holderCent = []
    holderPixelInd = []
    holderMerge = []
    holderSplit = []
    # Iterate through the current Frames
    for i in range(len(currentFrameList)):
        frame = currentFrameList[i]
        # check to see if frame is already in base
        if frame in frameList:
            continue

        startFrame = frameList[0]
        endFrame = frameList[-1]

        # If the frame is before the earliest frame in frameList
        # append all values to the start of lists. To make sure
        # we don't add things backwards, we create a holder
        # that we will then add to the start of the lits
        if frame < startFrame:
            lifetime += 1
            holderFrame.append(frame)
            holderSize.append(listFA.size[i])
            holderAvgInt.append(listFA.avgInt[i])
            holderIntInt.append(listFA.interInt[i])
            holderMaxInt.append(listFA.maxInt[i])
            holderMinInt.append(listFA.minInt[i])
            holderCent.append(listFA.centroid[i])
            holderPixelInd.append(listFA.pixelIndex[i])
            holderMerge.append(listFA.merge[i])
            holderSplit.append(listFA.split[i])

        # If the frame is after the lastest frame in the frameList
        # append all values to the end of lists
        if frame > endFrame:
            lifetime += 1
            frameList.append(frame)
            sizeList.append(listFA.size[i])
            averageIntensityList.append(listFA.avgInt[i])
            intergratedIntensityList.append(listFA.interInt[i])
            maxIntensityList.append(listFA.maxInt[i])
            minIntensityList.append(listFA.minInt[i])
            centroidList.append(listFA.centroid[i])
            pixelIndexList.append(listFA.pixelIndex[i])
            mergeList.append(listFA.merge[i])
            splitList.append(listFA.split[i])

    # Add all of our holder values to the start of the lists
    frameList = holderFrame + frameList
    sizeList = holderSize + sizeList
    averageIntensityList = holderAvgInt + averageIntensityList
    intergratedIntensityList = holderIntInt + intergratedIntensityList
    maxIntensityList = holderMaxInt + maxIntensityList
    minIntensityList = holderMinInt + minIntensityList
    centroidList = holderCent + centroidList
    pixelIndexList = holderPixelInd + pixelIndexList
    mergeList = holderMerge + mergeList
    splitList = holderSplit + splitList

    # It is possible that we are missing information between frames.
    # We need to create a new spot for this data and then collect it
    newFrameList = []
    newSizeList = []
    newAverageIntensityList = []
    newIntergratedIntensityList = []
    newMaxIntensityList = []
    newMinIntensityList = []
    newCentroidList = []
    newPixelIndexList = []
    newMergeList = []
    newSplitList = []

    # Lets go through the frames and see if any frame is missing
    startFrame = frameList[0]
    startIndex = frameList.index(startFrame)
    endFrame = frameList[-1]
    endIndex = frameList.index(endFrame)
    # Lets append the start information
    newFrameList.append(frameList[0])
    newSizeList.append(sizeList[0])
    newAverageIntensityList.append(averageIntensityList[0])
    newIntergratedIntensityList.append(intergratedIntensityList[0])
    newMaxIntensityList.append(maxIntensityList[0])
    newMinIntensityList.append(minIntensityList[0])
    newCentroidList.append(centroidList[0])
    newPixelIndexList.append(pixelIndexList[0])
    newMergeList.append(mergeList[0])
    newSplitList.append(splitList[0])
    for i in range(len(frameList)):
        # Skip past the first frame and last frame
        if i <= startIndex or i > endIndex:
            continue
        previousFrame = frameList[i - 1]
        trueNextFrame = previousFrame + 1
        currentFrame = frameList[i]
        # Check to see if the current Frame is 1+ the previous Frame
        # If it is, then we can just append normally
        if currentFrame == trueNextFrame:
            newFrameList.append(frameList[i])
            newSizeList.append(sizeList[i])
            newAverageIntensityList.append(averageIntensityList[i])
            newIntergratedIntensityList.append(intergratedIntensityList[i])
            newMaxIntensityList.append(maxIntensityList[i])
            newMinIntensityList.append(minIntensityList[i])
            newCentroidList.append(centroidList[i])
            newPixelIndexList.append(pixelIndexList[i])
            newMergeList.append(mergeList[i])
            newSplitList.append(splitList[i])
        # If this is not true, then we have missing Frames
        else:
            # Lets find how many frames are missing
            numMissingFrames = currentFrame - trueNextFrame
            # Lets estimate the centroid, as the average of the
            # previous and current centroid
            previousCentroid = centroidList[i - 1]
            currentCentroid = centroidList[i]
            predictedCentroid = (int((previousCentroid[0] + currentCentroid[0]) / 2.0),
                                 int((previousCentroid[1] + currentCentroid[1]) / 2.0))
            # We need to estimate the pixels that should have
            # been included. To be on the safe side, lets
            # Just take the intersection of pixels in the
            # previous frame and in the currentFrame
            previousPixelIndex = pixelIndexList[i - 1]
            currentPixelIndex = pixelIndexList[i]
            intersectionPixelIndex = intersection(previousPixelIndex, currentPixelIndex)
            # So now that we have the pixelIndexes, we need to get the PixelValues as well
            imgPixels = []
            trueFrame = video[trueNextFrame].copy()
            for pair in intersectionPixelIndex:
                imgPixels.append(trueFrame[pair[1], pair[0], :])
            # Now lets calculate all the values we need for this frame
            size = 0
            AvgIntensity = 0
            InterIntensity = 0
            # Go through each pixel in the focal adhesions box
            # in this frame. Toss out any 0s and keep all other
            # pixel values
            minIntensity = 99999999
            try:
                maxIntensity = np.mean(imgPixels[0].ravel())
            except:
                maxIntensity = 0
            for j in range(len(imgPixels)):
                temp = imgPixels[j].ravel()
                temp = np.mean(temp)
                if temp != 0:
                    size += 1
                    InterIntensity += temp
                    if temp > maxIntensity:
                        maxIntensity = temp
                    if temp < minIntensity:
                        minIntensity = temp
            # If the pixel was still alive in this frame and not noise
            # then lets update the values
            if size > 0:
                AvgIntensity = InterIntensity / size
            # We will use the same values for all the missing frames
            for k in range(numMissingFrames):
                lifetime += 1
                newAverageIntensityList.append(AvgIntensity)
                newSizeList.append(size)
                newIntergratedIntensityList.append(InterIntensity)
                newMaxIntensityList.append(maxIntensity)
                newMinIntensityList.append(minIntensity)
                newFrameList.append(trueNextFrame + k)
                newCentroidList.append(predictedCentroid)
                newPixelIndexList.append(intersectionPixelIndex)
                newMergeList.append([])
                newSplitList.append([])

            # Also append the frame that it skips too
            newFrameList.append(frameList[i])
            newSizeList.append(sizeList[i])
            newAverageIntensityList.append(averageIntensityList[i])
            newIntergratedIntensityList.append(intergratedIntensityList[i])
            newMaxIntensityList.append(maxIntensityList[i])
            newMinIntensityList.append(minIntensityList[i])
            newCentroidList.append(centroidList[i])
            newPixelIndexList.append(pixelIndexList[i])
            newMergeList.append(mergeList[i])
            newSplitList.append(splitList[i])

    # Now that we have all of our values stored in the same list, lets now
    # create a new FA object to add to our 'true' FA list
    FA = FocalAdhesion(ID, lifetime,
                       newMaxIntensityList, newMinIntensityList, newIntergratedIntensityList,
                       newAverageIntensityList, newSizeList,
                       newCentroidList, color, newPixelIndexList,
                       newFrameList, newMergeList, newSplitList)
    return FA


def FASearch(FA, Dict, FAList):
    try:
        otherList = Dict[FA.id]
        if len(otherList) != 0:
            for tempFA in otherList:
                holderList = []
                holderList.append(tempFA)
                associatedList = FASearch(tempFA, Dict, FAList)
                totalList = associatedList + holderList
                FAList = FAList + totalList

        return FAList
    except:
        return FAList

def track(imageReplaceType , saveType, maskReplaceType, csvfileSaveName, maskFolder,
            TrackerImages, minPixelShare, frameRange,
            splitPercent, mergePercent, minLifespan, minSize):
    print('Started Tracking')
    imageReplaceType='.'+imageReplaceType.replace('.','')
    imageType = '*'+imageReplaceType
    saveType='.'+saveType.replace('.','')
    maskReplaceType = '.' + maskReplaceType.replace('.', '')
    maskType='*'+maskReplaceType
    csvfileSaveName=csvfileSaveName+'.csv'
    videoPath = '..\\GUI\\Raw_Images\\' + TrackerImages + '\\'
    maskPath = '..\\GUI\\Masks\\' + maskFolder + '\\'
    #Check to see if folders to save to exist. If not, create them
    if not os.path.exists('..\\GUI\\Masks\\' + maskFolder + '\\CSVFiles'):
        os.makedirs('..\\GUI\\Masks\\' + maskFolder + '\\CSVFiles')
    if not os.path.exists('..\\GUI\\Masks\\'+maskFolder+'\\Tracked'):
        os.makedirs('..\\GUI\\Masks\\'+maskFolder+'\\Tracked')
    if not os.path.exists('..\\GUI\\Masks\\'+maskFolder+'\\Tracked\\Raw'):
        os.makedirs('..\\GUI\\Masks\\'+maskFolder+'\\Tracked\\Raw')
    csvfileSavePath = '..\\GUI\\Masks\\' + maskFolder + '\\CSVFiles\\'
    TrackedSavePath='..\\GUI\\Masks\\'+maskFolder+'\\Tracked\\'
    unmodifiedTrackedSavePath='..\\GUI\\Masks\\'+maskFolder+'\\Tracked\\Raw\\'

    mergePercent = mergePercent
    splitPercent = splitPercent
    minPixelShare = minPixelShare
    frameRange = frameRange
    minLifespan= minLifespan
    minSize= minSize

    #Load in the masks for contouring
    masksContour= getImages(maskPath, maskType, maskReplaceType, contourSetting=True)
    #load in the masks for tracking
    masks= getImages(maskPath, maskType, maskReplaceType)
    #Load the real Images
    video=getImages(videoPath, imageType,imageReplaceType)
    #Normalize the video
    # video=video/np.mean(video)

    #We want to remove the extra black space that was added to the mask because
    #of the process of splitting up the image in to even squares to segment it.
    #Lets return it to normal size
    (l,h,w,c)=np.shape(video)
    for i in range(len(masks)):
        tempy=masks[i]
        tempy=tempy[0:h, 0:w, :]
        masks[i]=tempy
    for i in range(len(masksContour)):
        tempy=masksContour[i]
        tempy=tempy[0:h, 0:w]
        masksContour[i]=tempy


    #Initialize the centroid tracker and frame dimensions
    ct=CentroidTracker()

    #We are going to give each FA a unique color, so lets
    #create a dictonary and a set we will use
    colorSet=set()
    uniqueColor=set()
    #Add black to the uniqueColor set, so a FA will
    #never be black
    uniqueColor.add((0,0,0))
    colorDict=OrderedDict()

    #Loop over the frames from the video
    tracked=[]
    pixelDictList=[]
    for i in range(len(masks)):
        #Read the next frame from the video
        frame=masks[i].copy()
        contourMask=masksContour[i].copy()
        oriMask=masks[i].copy()
        trueFrame=video[i].copy()
        frameNumber=i

        #pass frame through object detection and intialize the list of
        #bounding box rectangles
        rects, FAPixels=detectObjects(contourMask)
        #update our centroid tracker using the computed set of bounding box
        #rectangles
        objects, pixles=ct.update(rects, FAPixels)

        #loop over the tracked objects
        #Create a dictonary to hold all the pixel values for this frame
        pixelDict=OrderedDict()
        for (objectID, pixelArray) in pixles.items():
            centroid=objects[objectID]
            if objectID not in colorSet:
                #If we have not already given a FA a color,
                #Give it a color and add the color to the
                #dictionary
                colorSet.add(objectID)
                (r,g,b) = list(np.random.choice(range(256), size=3))
                color=(r,g,b)
                #Make sure the colors are truly unique
                while color in uniqueColor:
                    (r,g,b) = list(np.random.choice(range(256), size=3))
                    color=(r,g,b)
                uniqueColor.add(color)
                colorDict[objectID]=color
            color=colorDict[objectID]

            #We get a list of y coords and X coords, but they are seperate
            #so we need to combine
            y=pixelArray[0]
            x=pixelArray[1]
            coordinates=[]
            for i in range(len(x)):
                pair=(x[i],y[i])
                coordinates.append(pair)
            maskPixels=[]
            imgPixels=[]
            for i in range(len(coordinates)):
                pair=coordinates[i]
                #Color Focal Adhesion to Repersent Tracking them
                frame[pair[1],pair[0], 0]=color[0]
                frame[pair[1],pair[0], 1]=color[1]
                frame[pair[1],pair[0], 2]=color[2]
                #Get the pixels in the mask and in the original image
                maskPixels.append(oriMask[pair[1],pair[0], :])
                imgPixels.append(trueFrame[pair[1],pair[0], :])
            FAPixels=np.multiply(imgPixels,maskPixels)
            FAPixels=FAPixels[:,0]
            package=(centroid, coordinates, color, frameNumber)
            pixelDict[objectID]=(FAPixels, package)
        tracked.append(frame)
        pixelDictList.append(pixelDict)
        #Loop over each tracked object and extract the pixel values from be box
        #from the real image, including only pixels that are white in the mask


    #Now we want to create a dictonary for each Focal Adhesion that contains only
    #The pixles related to that focal adheision throughout the entire time

    #To know if we already have an existing dictonary, lets create
    #a set to track all object IDs that already have been seen
    seenIds=set()
    #Create a dictonary to hold all Focal adhesion dictonaries
    masterFADict=OrderedDict()
    for i in range(len(pixelDictList)):
        frame=pixelDictList[i]
        for (objectID, FA) in frame.items():
            #check to see if the current objectID is in the set
            if objectID not in seenIds:
                #add the objectID to the seenIds set
                seenIds.add(objectID)
                #Create a new dictonary for this focal adhesion
                currentFA=OrderedDict()
                #Add the new dictonary to the master dictonary
                #of Focal adheisions
                masterFADict[objectID]=currentFA
            #get the Focal Adhesion Dictonary for the current Focal
            #Adhesion
            currentFA=masterFADict[objectID]
            #Add the pixel values for the current frame
            #To the current Focal Adhesion's dictonary
            currentFA[i]=(FA[0].ravel(),FA[1])
            #Update the dictonary in the master dictonary
            masterFADict[objectID]=currentFA

    #So now the dictonary we have is organized by individual
    #Focal adhesion tracked throughout the video.
    #However, We don't want to just return a bunch of pixel values.
    #So lets make some calculations on the pixels and then
    #stick them into a new dictonary, and then stick that
    #dictonary into a list
    masterFAInfo=[]
    for (objectID, FADict) in masterFADict.items():
        FAInfo=OrderedDict()
        focalAdhesionID=objectID
        lifeTime=0
        AvgIntList=[]
        SizeList=[]
        InterIntList=[]
        MaxIntList=[]
        MinIntList=[]
        centroid=[]
        pixelIndex=[]
        frameNumber=[]
        mergeList=[]
        splitList=[]
        #Go through each frame that in which the focal adhesion
        #is alive
        for (objID, FAFrame) in FADict.items():
            frame=FAFrame[0]

            package=FAFrame[1]
            color=package[2]
            size=0
            AvgIntensity=0
            InterIntensity=0
            #Go through each pixel in the focal adhesions box
            #in this frame. Toss out any 0s and keep all other
            #pixel values
            minIntensity=99999999
            try:
                maxIntensity=frame[0]
            except:
                maxIntensity=0
            for i in range(len(frame)):
                if frame[i]!=0:
                    size+=1
                    InterIntensity+=frame[i]
                    if frame[i]>maxIntensity:
                        maxIntensity=frame[i]
                    if frame[i]<minIntensity:
                        minIntensity=frame[i]
            #If the pixel was still alive in this frame and not noise
            #then lets update the values

            if size>3:
                AvgIntensity=InterIntensity/size
                lifeTime+=1
                AvgIntList.append(AvgIntensity)
                SizeList.append(size)
                InterIntList.append(InterIntensity)
                MaxIntList.append(maxIntensity)
                MinIntList.append(minIntensity)
                frameNumber.append(package[3])
                centroid.append(package[0])
                pixelIndex.append(package[1])
                mergeList.append([])
                splitList.append([])



        #Finally, Lets create a Focal Adhesion Object that we can
        #store in our masterFAInfo list
        FA=FocalAdhesion(focalAdhesionID, lifeTime,
                         MaxIntList, MinIntList, InterIntList,
                         AvgIntList, SizeList,
                         centroid, color, pixelIndex,
                         frameNumber, mergeList, splitList)
        #Only keep focal Adhesions that are alive for more then 0 frames
        if lifeTime>0:
            masterFAInfo.append(FA)

    #So now we have all the information about each Focal Adhesion
    #that we want. When we print this out into an spreadsheet
    #we will want the focal adhesions that lived the longest
    #to be at the top of the sheet, so lets go ahead and sort
    #the list now to make things easier later on
    def FASorter(FA):
        return FA.id
    masterFAInfo.sort(key=FASorter, reverse = False)



    #Lets go through all the Focal Adhesions and try
    #and correct any times we think that an FA filckered
    #or incorrectly merged and then unmerged


    #Set minimum sharing to be considerd the same FA

    #Create a dictonary that will contain a list
    #of all FAs that should be associated together
    #for a given id
    associatedFADict=OrderedDict()
    #Create a set of all the combinations that we have already seen
    combinationSet=set()

    #Go through all the FAs
    for FA in masterFAInfo:
        #Make a new associated list for the current FA
        associatedList=[]
        #Add the current FA and list to the dict
        associatedFADict[FA.id]=associatedList
        for erFA in masterFAInfo:
            combinationID1=(FA.id,erFA.id)
            combinationID2=(erFA.id,FA.id)
            if combinationID1 in combinationSet or combinationID2 in combinationSet:
                continue
            combinationSet.add(combinationID1)
            combinationSet.add(combinationID2)
            #If the two FAs that we are looking at
            #share the same id, then we don't want to
            #compare them, because they will be the exact same
            if FA.id==erFA.id:
                continue
            #For each frame, check to see if the
            #they share some minimum overlap with
            #each other with in some frame range
            for i in range(FA.lifetime):
                #get a frame number
                frameNum=FA.frameNumber[i]
                for j in range(erFA.lifetime):
                    erframeNum=erFA.frameNumber[j]
                    #If they coexist in the same frame,
                    #then they can't possibley be the same
                    #FA, so skip over comparision
                    if frameNum==erframeNum:
                        continue
                    #If the frames are within the given range of
                    #each other, then lets compare the pixel
                    #indxes to how much overlap they have

                    if (frameNum-frameRange)<=erframeNum and erframeNum<=(frameNum+frameRange):
                        pixlesIDX=FA.pixelIndex[i]
                        pixelLen=len(pixlesIDX)
                        erpixlesIDX=erFA.pixelIndex[j]
                        erpixelLen=len(erpixlesIDX)
                        #We will set the bigger size
                        total=max(pixelLen,erpixelLen)
                        #see how many pixels they share
                        numShared=0
                        for index in pixlesIDX:
                            if index in erpixlesIDX:
                                numShared+=1
                        #calculate percent pixel share
                        percentShared=numShared/total
    #                     if percentShared>0.0:
    #                         print(percentShared)
                        #percentShared is equal to or greater than out min percent shared
                        #then we will say these are the same Focal Adhesions. We will
                        #make append this erroneous FA to a list of erroneous FAs
                        #that should be associated with this current FA. We will also
                        #skip looking at any more frames
                        if percentShared>=minPixelShare:
                            associatedFADict[FA.id].append(erFA)
                            continue
                    

    #Delete duplicates in List
    for (objectID, aList) in associatedFADict.items():
        if len(aList)!=0:
            #delete duplicates
            tempSet=set()
            tempList=list()
            for FA in aList:
                if FA.id not in tempSet:
                    tempSet.add(FA.id)
                    tempList.append(FA)
            associatedFADict[objectID]=tempList


    #Now that we found all the FAs that need to be combined
    #Lets combine them to get all the 'true' focal adhesions

    #Make a list to store the 'True' FAs in
    trueFAList=[]
    #We need to make sure we don't re-add any FAs that we combined
    #so make a set of ids that were used
    usedSet=set()
    errorSet=set()

    for FA in masterFAInfo:
        #If the FA has already been used, then skip it
        #so that it is not added twice
        if FA.id in usedSet:
            continue
        #add FA.id to usedSet
        usedSet.add(FA.id)
        #Get the associated list for the current FA
        aList=associatedFADict[FA.id]
        #If the FA does not have any associated FAs
        #Then we can just directly add it to the trueFAList
        if len(aList)==0:
            trueFAList.append(FA)
            continue

        #We will need to create a new FA object that will contain
        #all the frames that the FA is actually alive for
        #without including any duplicate frames
        else:
            usedSet.add(FA.id)
            for listFA in aList:
                if listFA.id in usedSet:
                    continue
                #Ignore duplicate
                if FA.id==listFA.id:
                    continue
                #Find out if listFA.id is needs to be concatenated
                #with another FA before it can be concatenated
                #with this FA. Create a new function to do this

                #Lets go through our listFA and see if there are any
                #FAs that it should be combined with before we combine
                #this listFA with FA
                FAList=[]
                associatedList=FASearch(listFA, associatedFADict, FAList)
                for tempFA in associatedList:
                    if tempFA.id==FA.id:
                        continue
                    usedSet.add(tempFA.id)
                    listFA=joinFAs(listFA, tempFA, video)

                #Add all FAs the list of used ids
                usedSet.add(listFA.id)
                FA=joinFAs(FA, listFA, video)


            trueFAList.append(FA)


    # #Some FAs that were used cannot be properly
    # #deleted for some reason, so lets just mark them
    # #with an flag saying that they are potentially
    # #erroneous
    # for FA in trueFAList:
    #     if FA.id in usedSet:
    #         FA.errorFlag=True


    #Now Some FA may be sharing Pixels when they should not be
    #Lets find these sharing FAs and take away from the
    #Bigger FA that is sharing
    for FA in trueFAList:
        for otherFA in trueFAList:
            if FA.id==otherFA.id:
                continue
            #Lets find the frames in which they coexist
            sharedFrames=intersection(FA.frameNumber, otherFA.frameNumber)
            #if they don't share any frames, continue
            if len(sharedFrames)==0:
                continue
            for i in range(len(sharedFrames)):
                frameNumber=sharedFrames[i]
                index=FA.frameNumber.index(frameNumber)
                otherIndex=otherFA.frameNumber.index(frameNumber)
                pixles=FA.pixelIndex[index]
                otherpixles=otherFA.pixelIndex[otherIndex]
                #find the intersection of thier pixels
                sharedPixels=intersection(pixles,otherpixles)
                #If they don't share any frames continue
                if len(sharedPixels)==0:
                    continue
                #If they share pixels, delete shared pixels from
                #the larger FA
                if len(pixles)>len(otherpixles):
                    for pixel in sharedPixels:
                        pixles.remove(pixel)
                    #Recalulate all the FA's value
                    imgPixels=[]
                    trueFrame=video[frameNumber].copy()
                    for pair in sharedPixels:
                        imgPixels.append(trueFrame[pair[1],pair[0], :])
                    #Now lets calculate all the values we need for this frame
                    size=0
                    AvgIntensity=0
                    InterIntensity=0
                    #Go through each pixel in the focal adhesions box
                    #in this frame. Toss out any 0s and keep all other
                    #pixel values
                    minIntensity=99999999
                    try:
                        maxIntensity=np.mean(imgPixels[0].ravel())
                    except:
                        maxIntensity=0
                    for j in range(len(imgPixels)):
                        temp=imgPixels[j].ravel()
                        temp=np.mean(temp)
                        if temp!=0:
                            size+=1
                            InterIntensity+=temp
                            if temp>maxIntensity:
                                maxIntensity=temp
                            if temp<minIntensity:
                                minIntensity=temp
                    if size>0:
                        AvgIntensity=InterIntensity/size
                    FA.size[index]=size
                    FA.avgInt[index]=AvgIntensity
                    FA.maxInt[index]=maxIntensity
                    FA.minInt[index]=minIntensity
                    FA.interInt[index]=InterIntensity
                    FA.pixelIndex[index]=pixles

                elif len(otherpixles)>len(pixles):
                    for pixel in sharedPixels:
                        otherpixles.remove(pixel)
                        #Recalulate all the FA's value
                    imgPixels=[]
                    trueFrame=video[frameNumber].copy()
                    for pair in sharedPixels:
                        imgPixels.append(trueFrame[pair[1],pair[0], :])
                    #Now lets calculate all the values we need for this frame
                    size=0
                    AvgIntensity=0
                    InterIntensity=0
                    #Go through each pixel in the focal adhesions box
                    #in this frame. Toss out any 0s and keep all other
                    #pixel values
                    minIntensity=99999999
                    try:
                        maxIntensity=np.mean(imgPixels[0].ravel())
                    except:
                        maxIntensity=0
                    for j in range(len(imgPixels)):
                        temp=imgPixels[j].ravel()
                        temp=np.mean(temp)
                        if temp!=0:
                            size+=1
                            InterIntensity+=temp
                            if temp>maxIntensity:
                                maxIntensity=temp
                            if temp<minIntensity:
                                minIntensity=temp
                    if size>0:
                        AvgIntensity=InterIntensity/size
                    otherFA.size[otherIndex]=size
                    otherFA.avgInt[otherIndex]=AvgIntensity
                    otherFA.maxInt[otherIndex]=maxIntensity
                    otherFA.minInt[otherIndex]=minIntensity
                    otherFA.interInt[otherIndex]=InterIntensity
                    otherFA.pixelIndex[otherIndex]=otherpixles


    #Now that we have dealt with any flickering or improper merging
    #lets address real merging and spliting.
    #To determine if two FAs have merged, lets look at the end of every
    #FAs life. If the location that the FA died shares some percentage
    #of pixels with a FA in the next Frame, then the two FAs must have mearged
    #To determine if a FA has split in two

    #Set the percent overlap for a merge

    for FA in trueFAList:
            #Find out when the FA died
            death=FA.frameNumber[-1]
            deathIdx=FA.frameNumber.index(death)
            preDeathPixels=FA.pixelIndex[-1]
            #Find out when the FA was born
            birth=FA.frameNumber[0]
            birthIdx=FA.frameNumber.index(birth)
            postBirthPixels=FA.pixelIndex[0]
            #Compare to all other FAs
            for otherFA in trueFAList:
                #Skip self
                if FA.id==otherFA.id:
                    continue
                #See it this other FA was still alive in the
                #frame directly after FA died
                postDeath=death+1
                if (postDeath) in otherFA.frameNumber:
                    #If the other FA is alive in the next Frame
                    #then we need to check if there was the required
                    #over lap with FA to be considered a merge
                    otherIdx=otherFA.frameNumber.index(postDeath)
                    otherPixels=otherFA.pixelIndex[otherIdx]
                    numShared=0
                    for index in preDeathPixels:
                            if index in otherPixels:
                                numShared+=1
                    percentShared=(numShared)/len(preDeathPixels)
                    #If they shared the min percent of pixels, then
                    #we want to note that they merged together in both
                    #of their merge properties
                    if percentShared>=mergePercent:
                        otherFA.merge[otherIdx].append(FA.id)
                        FA.merge[deathIdx].append(otherFA.id)
                #See if this other FA was alive in the frame directly
                #before FA was born
                preBirth=birth-1
                if (preBirth) in otherFA.frameNumber:
                    #If the other FA is alive in the previous Frame
                    #then we need to check if there was the required
                    #over lap with FA to be considered a merge
                    otherIdx=otherFA.frameNumber.index(preBirth)
                    otherPixels=otherFA.pixelIndex[otherIdx]
                    numShared=0
                    for index in postBirthPixels:
                            if index in otherPixels:
                                numShared+=1
                    percentShared=(numShared)/len(postBirthPixels)
                    #If they shared the min percent of pixels, then
                    #we want to note that they merged together in both
                    #of their merge properties
                    if percentShared>=splitPercent:
                        otherFA.split[otherIdx].append(FA.id)
                        FA.split[birthIdx].append(otherFA.id)

    #NOTE: If a split is on the first frame of an FA's life,
    #This means that it split from another FA between the previous Frame and now

    #NOTE: If a split occurs on any other frame, Then this means
    #That the FA splits between now and the next Frame

    #NOTE: If a merge is on the last frame of an FA's life,
    #This means that that it merges with another FA in between
    #now and the next Frame

    #NOTE: IF a merge occurs on any other frame, Then this means
    #That the FA merged between the last frame and now


    #Since the lifetime may have changed, resort the FAs
    def FASorter(FA):
        return FA.lifetime
    trueFAList.sort(key=FASorter, reverse = True)


    #Now that we have addresed various issues, lets re color the masks
    newTracked=[]
    for i in range(len(masks)):
        #Make a blank canvas the size of the mask
    #     frame=np.zeros_like(masks[i])
        frame=masks[i].copy()
        #Find all focal adhesions in the current frame and color them
        for FA in trueFAList:
            #Check to see if error Flag is raised. If so, don't draw
    #         if FA.errorFlag==True:
    #             continue
            #Check to see if Focal Adhesion exits in current frame
            if i in FA.frameNumber:
                #find the index associated with this frame
                idx=FA.frameNumber.index(i)
                color=FA.color
                coordinates=FA.pixelIndex[idx]
                #Color the relevant pixles
                for j in range(len(coordinates)):
                    pair=coordinates[j]
                    #Color Focal Adhesion to Repersent Tracking them
                    frame[pair[1],pair[0], 0]=color[0]
                    frame[pair[1],pair[0], 1]=color[1]
                    frame[pair[1],pair[0], 2]=color[2]
    #     print(count)
        newTracked.append(frame)


    #Now lets take out Focal adhesions and put them in a csv file.
    #Each Focal adhesion will be its own block within the csv file

    #Define the csv file that we will be creating/editing
    csvfile=csvfileSavePath+csvfileSaveName

    #Delete the old one if its there
    try:
        os.remove(csvfile)
    except:
        print('Creating New file')

    #Define the header that we will be using numerous times
    header=['Focal_Adhesion_ID', 'Color', 'Frame Number',
            'Pixel_Area', 'Average_Intensity',
            'Intergrated_Intensity','Maximum_Intensity',
            'Minimum_Intensity', 'Centroid',
            'Merge', 'Split', 'Error Flag','Pixel_Indexes']

    with open(csvfile,'w') as file:
        writer=csv.writer(file)
        for FA in trueFAList:
            #Create a new header whenever we add a new Focal Adhesion
            writer.writerow(header)
            for i in range(FA.lifetime):
                row=[FA.id, FA.color, FA.frameNumber[i],
                     FA.size[i], FA.avgInt[i],
                     FA.interInt[i], FA.maxInt[i],
                     FA.minInt[i], FA.centroid[i],
                     FA.merge[i], FA.split[i], FA.errorFlag, FA.pixelIndex[i]]
                writer.writerow(row)
        #Close the file we are writing to
        file.close()

    saver(newTracked, saveType, TrackedSavePath)
    saver(tracked, saveType, unmodifiedTrackedSavePath)
    print('Finished Tracking')
