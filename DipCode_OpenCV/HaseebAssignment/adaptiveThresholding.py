from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv
import glob



def fileCollector(path):
    loadedImages = []
    files = glob.glob(path)
    for i in files:
        img = cv.imread(i, cv.IMREAD_GRAYSCALE)
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
adaptiveImgs=[]

adaptiveImgOut='C:/Users/walee/Documents/DipCode/HaseebAssignment/AdaptiveOut'
adaptiveCSV='C:/Users/walee/Documents/DipCode/HaseebAssignment/AdaptiveOut.csv'

for i in range(0,len(allImgNames),1):
    img=allImgs[i]
    ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    img=cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,2)
    adaptiveImgs.append(img)

confMatrixforALL=[[],[],[],[]]
confMatrixforALL= createOutputImgs(adaptiveImgOut,allImgNames,adaptiveImgs,segmentedImgs,confMatrixforALL)
fileWriter(adaptiveCSV,allImgNames,confMatrixforALL)