from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv
import glob



def fileCollector(path):
    loadedImages = []
    files = glob.glob(path)
    for i in files:
        img = cv.imread(i, cv.IMREAD_COLOR)
        loadedImages.append(img)
    return loadedImages, files

def fileWriter(csvFilePath, inIms, confMat):
    with open(csvFilePath, 'w', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        row = ['Name','Dice Coefficient']
        csv_writer.writerow(row)
        row = []
        for i in range(1, len(inIms), 1):
            newName = (inIms[i].split("\\", 1)[1])
            row = [newName, diceCoff(confMat[3][i], confMat[2][i], confMat[1][i])]
            csv_writer.writerow(row)
            row = []

def diceCoff(TP, FN, FP):
    return (2 * TP) / (FN + (2 * TP) + FP + FP)

def myConfusionMatrix(myThresh, OriginalThresh):
    TP = 0
    FP = 0
    FN = 0
    TN=0
    y=[]
    for i in range(1, myThresh.shape[0], 1):
        for j in range(1, myThresh.shape[1], 1):
            if myThresh[i][j] == 255 and OriginalThresh[i][j]==255:
                TP = TP + 1
            elif myThresh[i][j] ==0 and OriginalThresh[i][j]==255:
                FN = FN + 1
            elif myThresh[i][j] == 255 and OriginalThresh[i][j]==0:
                FP = FP + 1
            elif myThresh[i][j] == 0 and OriginalThresh[i][j]==0:
                TN=TN+1

    print(TN,FP,FN,TP)
    return (TN, FP, FN, TP)


def createOutputImgs(outPath, inIms, CC_Imgs, segmentedOriginals, confMatrixForAll):
    for i in range(0, len(inIms), 1):
        newName = (inIms[i].split("\\", 1)[1])
        TN, FP, FN, TP=myConfusionMatrix(CC_Imgs[i], segmentedOriginals[i])
        confMatrixForAll[0].append(TN)
        confMatrixForAll[1].append(FP)
        confMatrixForAll[2].append(FN)
        confMatrixForAll[3].append(TP)
        print(f"{outPath}" + f"/{newName}")
        cv.imwrite(f"{outPath}" + f"/{newName}", CC_Imgs[i])
    return confMatrixForAll

loadedColoredImgsPath = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/Dermoscopic_Image/*.*'
loadedOriginalSegmented = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/lesion/*.*'


allImgs, allImgNames = fileCollector(loadedColoredImgsPath)  # all Colored Images loaded in greyScale
segmentedImgs, _ = fileCollector(loadedOriginalSegmented)
regionImgs=[]

regionImgOut='C:/Users/walee/Documents/DipCode/HaseebAssignment/RegionOut'
regionCSV='C:/Users/walee/Documents/DipCode/HaseebAssignment/RegionOut.csv'
kernel = np.ones((3,3),np.uint8)
for i in range(0,len(allImgNames),1):
    img=allImgs[i]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    img=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    ret,img=cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    img=img.astype(np.uint8)
    regionImgs.append(img)


confMatrixforALL=[[],[],[],[]]
confMatrixforALL= createOutputImgs(regionImgOut,allImgNames,regionImgs,segmentedImgs,confMatrixforALL)
fileWriter(regionCSV,allImgNames,confMatrixforALL)