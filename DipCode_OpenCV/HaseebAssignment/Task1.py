from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from CC import ConnectedComponents
import glob
import csv



def diceCoff(TP, FN, FP):
    return (2 * TP) / (FN + (2 * TP) + FP + FP)


def sensitivity(TP, FN):
    return TP / (TP + FN)


def specificity(FP, TN):
    return FP / (TN + FP)


def fileCollector(path):
    loadedImages = []
    files = glob.glob(path)
    for i in files:
        img = cv.imread(i, cv.IMREAD_GRAYSCALE)
        loadedImages.append(img)
    return loadedImages, files


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



def preprocessing(img):
    newIm = img.copy()
    cv.GaussianBlur(img, (7, 7), 1.2, newIm)
    _, newIm = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    return newIm


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

def dimeter(img):
    y=[]
    count=0
    for i in range(0,img.shape[0],1):
        for j in range(0, img.shape[1], 1):
            if img[i][j]==255:
                count=count+1
        y.append(count)
    x=int(max(y)/1000)
    return x


def fileWriter(csvFilePath, inIms, confMat):
    with open(csvFilePath, 'w', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        row = ['ImgName','Dice Coefficient', 'Sensitivity', 'Specificity','Diameters']
        csv_writer.writerow(row)
        row = []
        for i in range(1, len(inIms), 1):
            newName = (inIms[i].split("\\", 1)[1])
            row = [newName, diceCoff(confMat[3][i], confMat[2][i], confMat[1][i]),
                   sensitivity(confMat[3][i], confMat[2][i]), specificity(confMat[1][i], confMat[0][i]),allDiameter[i]]
            csv_writer.writerow(row)
            row = []


loadedColoredImgsPath = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/Dermoscopic_Image/*.*'
outputImgsPath = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/CC_LabellingResults'
loadedOriginalSegmented = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/lesion/*.*'
loadedMySegmented = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/CC_LabellingResults/*.*'
OutputCSV = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/output.csv'

allImgs, allImgNames = fileCollector(loadedColoredImgsPath)  # all Colored Images loaded in greyScale
segmentedImgs, _ = fileCollector(loadedOriginalSegmented)
CCImgs = []
allDiameter=[]
count=0
for i in range(0, len(allImgNames), 1):
    img = preprocessing(allImgs[i])
    connObj = ConnectedComponents(img)
    var=connObj._4Connectivity().astype(np.uint8)
    _,var=cv.threshold(img,127,255,cv.THRESH_BINARY)
    CCImgs.append(var)
    allDiameter.append(dimeter(var))
    print('Applying CC'+str(count))
    count=count+1
confMatrixforALL = [[],[],[],[]]
confMatrixforALL = createOutputImgs(outputImgsPath, allImgNames, CCImgs, segmentedImgs, confMatrixforALL)
fileWriter(OutputCSV,allImgNames ,confMatrixforALL)

cv.waitKey()
cv.destroyAllWindows()