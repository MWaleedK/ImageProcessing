from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
from threading import *
from queue import Queue

MAX = 256
mutex=Lock()
path = 'C:/Users/walee/Documents/DipCode/Assignments/Specimens/*.*'


def gettingMaxRepulsive(path):
    testFiles = glob.glob(path)

    vSetNaught = []
    for image in testFiles:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # assuming the lower 20 percentile pixels would be background
        minPer = np.percentile(img, 20)
        vSetNaught.append(minPer)
    # creating the vSet and returning it
    repulsiveMax = int(max(vSetNaught))
    return (np.array([range(repulsiveMax + 1, 256, 1)])).astype(np.uint8)


# C:/Users/walee/Documents/DipCode/Assignments/Specimens/1_left.jpeg
# C:/Users/walee/Documents/DipCode/Lab3
def _8Connectivity(img,vSet):#input image and vSet
    img = np.lib.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=(0, 0))
    rows = img.shape[0]
    columns = img.shape[1]
    counter = 0
    label = [0]
    newImg = np.zeros([rows, columns], np.uint8)
    # time.sleep(2)
    for i in range(1, rows, 1):
        for j in range(1, columns, 1):
            if np.in1d(img[i][j], vSet):
                if (np.in1d(img[i][j - 1], vSet, assume_unique=False, invert=True) and np.in1d(img[i - 1][j], vSet,
                                                                                               assume_unique=False,
                                                                                               invert=True) and np.in1d(
                    img[i - 1][j - 1], vSet, assume_unique=False, invert=True) and np.in1d(img[i - 1][j + 1], vSet,
                                                                                           assume_unique=False,
                                                                                           invert=True)):
                    counter = counter + 1
                    label.append(counter)
                    newImg[i][j] = counter
                elif np.in1d(img[i - 1][j - 1], vSet, assume_unique=False, invert=False):
                    newImg[i][j] = newImg[i - 1][j - 1]
                elif np.in1d(img[i][j - 1], vSet, assume_unique=False, invert=False):
                    newImg[i][j] = newImg[i][j - 1]
                elif np.in1d(img[i - 1][j + 1], vSet, assume_unique=False, invert=False):
                    newImg[i][j] = newImg[i - 1][j + 1]
                elif np.in1d(img[i - 1][j], vSet, assume_unique=False, invert=False):
                    newImg[i][j] = newImg[i - 1][j]

                if (np.in1d(img[i - 1][j], vSet, assume_unique=False, invert=False) and np.in1d(img[i][j - 1], vSet,
                                                                                                assume_unique=False,
                                                                                                invert=False)):
                    temp = np.array([newImg[i - 1][j], newImg[i][j - 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin
                if (np.in1d(img[i - 1][j], vSet, assume_unique=False, invert=False) and np.in1d(img[i - 1][j - 1], vSet,
                                                                                                assume_unique=False,
                                                                                                invert=False)):
                    temp = np.array([newImg[i - 1][j], newImg[i - 1][j - 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin
                if (np.in1d(img[i - 1][j], vSet, assume_unique=False, invert=False) and np.in1d(img[i - 1][j + 1],
                                                                                                vSet,
                                                                                                assume_unique=False,
                                                                                                invert=False)):
                    temp = np.array([newImg[i - 1][j], newImg[i - 1][j + 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin
                if (np.in1d(img[i][j - 1], vSet, assume_unique=False, invert=False) and np.in1d(img[i - 1][j + 1],
                                                                                                vSet,
                                                                                                assume_unique=False,
                                                                                                invert=False)):
                    temp = np.array([newImg[i][j - 1], newImg[i - 1][j + 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin
                if (np.in1d(img[i][j - 1], vSet, assume_unique=False, invert=False) and np.in1d(img[i - 1][j - 1],
                                                                                                vSet,
                                                                                                assume_unique=False,
                                                                                                invert=False)):
                    temp = np.array([newImg[i][j - 1], newImg[i - 1][j - 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin
                if (np.in1d(img[i - 1][j + 1], vSet, assume_unique=False, invert=False) and np.in1d(img[i - 1][j - 1],
                                                                                                    vSet,
                                                                                                    assume_unique=False,
                                                                                                    invert=False)):
                    temp = np.array([newImg[i - 1][j + 1], newImg[i - 1][j - 1]])
                    tempMin = np.amin(temp)
                    label[np.amax(temp)] = tempMin
                    newImg[i][j] = tempMin

    for i in range(1, rows, 1):
        for j in range(1, columns, 1):
            if (np.in1d(img[i][j], label)):
                newVal = label[newImg[i][j]]
                newImg[i][j] = newVal

    return newImg.astype(np.uint8), label


#multithreading for object detection
def sideToSide(pImage,directions,threadHandler,xCenter,yCenter,checkVar,Q):
        if(threadHandler==0):
            for i in range(1, pImage.shape[0], 1):
                variance=0#lets say a couple of connected blocks occur in the image where they shouldn't (artifact), so this variance would allow propagation till we reach the original object
                if(pImage[i][yCenter]!=checkVar):
                    directions=directions+1
                    #variance=0
                else:
                    '''if(variance<8):
                        directions=directions+1
                        variance=variance+1
                    else:
                        if(directions>8):
                            directions=directions-variance'''
                    break #no use continuing if variance starts counting original CC object of interest... Subtract variance and get co-ordinates
            mutex.acquire()
            Q.put((directions, yCenter, threadHandler))
            mutex.release()
        elif(threadHandler==1):
            myRange=pImage.shape[0]-1
            for i in range(myRange, 0, -1):
                variance = 0  # lets say a couple of connected blocks occur in the image where they shouldn't (artifact), so this variance would allow propagation till we reach the original object
                if (pImage[i][yCenter] != checkVar):
                    directions = directions + 1
                    #variance=0
                else:
                    '''if (variance < 8):
                        directions = directions + 1
                        variance = variance + 1
                    else:
                        if (directions > 8):
                            directions = directions - variance'''
                    break  # no use continuing if variance starts counting original CC object of interest... Subtract variance and get co-ordinates
            directions=myRange-directions
            mutex.acquire()
            Q.put((directions,yCenter, threadHandler))
            mutex.release()

        elif(threadHandler==2):
            myRange=pImage.shape[1]-1
            for j in range(1,myRange, 1):
                variance = int(0)  # lets say a couple of connected blocks occur in the image where they shouldn't (artifact), so this variance would allow propagation till we reach the original object
                if (pImage[xCenter][j] != checkVar):
                    directions = directions + 1
                    #variance=0
                else:
                    '''if (variance < 8):
                        directions = directions + 1
                        variance = variance + 1
                    else:
                        if (directions > 8):
                            directions = directions - variance'''
                    break  # no use continuing if variance starts counting original CC object of interest... Subtract variance and get co-ordinates
            mutex.acquire()
            Q.put((xCenter, directions, threadHandler))
            mutex.release()

        elif(threadHandler==3):
            myRange=pImage.shape[1]-1
            for j in range(myRange, 0, -1):
                variance = 0  # lets say a couple of connected blocks occur in the image where they shouldn't (artifact), so this variance would allow propagation till we reach the original object
                if (pImage[xCenter][j] != checkVar):
                    directions = directions + 1
                    #variance=0
                else:
                    '''if (variance < 8):
                        directions = directions + 1
                        variance = variance + 1
                    else:
                        if (directions > 8):
                            directions = directions - variance'''
                    break  # no use continuing if variance starts counting original CC object of interest... Subtract variance and get co-ordinates
            directions = myRange - directions
            mutex.acquire()
            Q.put((xCenter,directions,threadHandler))
            mutex.release()




def getBiggestCCObject(originalImg,myImage):#myImage is the CC analysis image
    #hist = np.bincount(myImage.ravel(), minlength=256)
    hist=np.zeros(np.amax(myImage)+1, np.uint32)
    IMGrows=myImage.shape[0]
    IMGCols=myImage.shape[1]
    for i in range(1, IMGrows, 1):
        for j in range(1, IMGCols, 1):
            hist[myImage[i][j]]=hist[myImage[i][j]]+1
    biggestObjectLoc=(np.argmax(hist[1:]))+1 #+1 because we're neglecting index 0 as it has the most pixel count i.e background
    biggestObjectCellCount = hist[biggestObjectLoc]
    xCenter=int(IMGrows/2)
    yCenter=int(IMGCols/2)
    dirSize=4
    direction=np.zeros(dirSize,dtype=np.uint32)#4-directions... Top to bottom, Bottom to Top, Left to Right, Right to Left
    #multithreading in four directions
    print('ThreadingGo')
    #for Left Side:(Ycenter,x++)=(column,row++)
    myThreads=[]
    threadHandler=int(0)
    Q = Queue()# contains co-ordinates of 4 points in cartesian co-ordinate manner (x,y)... Use wisely

    for thread in range(0,dirSize,1):

        t=(Thread(target=sideToSide, args=(myImage, direction[thread],threadHandler,xCenter,yCenter,biggestObjectLoc,Q)))
        t.start()
        threadHandler=threadHandler+1
        myThreads.append(t)

    for thread in myThreads:
        thread.join()

    orderedQ = [0, 0, 0, 0]  # bigger Images might mess up Q's order, hence this... The orientation is the same as Q...Cartesian coordinate system
    for  i in range(0,dirSize,1):
        val=Q.get()
        orderedQ[val[2]]=(val[0],val[1])
    print('ThreadingEnd')
    #you have the co-ordinates now, Crop Image

    print("orderedQ")
    print(orderedQ)


    return originalImg[orderedQ[2][1]:orderedQ[3][1],orderedQ[0][0]:orderedQ[1][0]]


origImg=cv2.imread('C:/Users/walee/Downloads/circle1.jpeg', cv2.IMREAD_GRAYSCALE)
myImage, label = _8Connectivity()
cropIm=getBiggestCCObject(origImg)
cv2.imshow('crop',cropIm)
cv2.imshow('newImg', myImage)
cv2.waitKey()
cv2.destroyAllWindows()