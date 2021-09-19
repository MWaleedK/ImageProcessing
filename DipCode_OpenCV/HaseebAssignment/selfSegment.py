from cv2 import cv2 as cv
import numpy as np
import time

class regionExtraction:
    def __init__(self,img):
        self.img=img
        self.rows=img.shape[0]
        self.columns=img.shape[1]
    
    def differentiation(self):
        contours=[[],[]]
        for i in range(0,self.rows-1,1):
            for j in range(0,self.columns-1,1):
                if(j<self.columns):
                    val=self.img[i][j+1]-self.img[i][j]
                    if val!=0:
                        contours[0].append(i)
                        contours[1].append(j)

        print(contours)
        return contours
    
    def segmentOut(self,img,contours):
        for i in range(0,len(contours[0]),1):
            x=contours[0][i]
            y=contours[1][i]
            img[x][y]=np.array([255,0,0])
        return img

loadedColoredImgsPath = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/Dermoscopic_Image/*.*'
outputImgsPath = 'C:/Users/walee/Documents/DipCode/HaseebAssignment/CC_LabellingResults'
imgCol=cv.imread('C:/Users/walee/Documents/DipCode/HaseebAssignment/Dermoscopic_Image/IMD002.bmp',cv.IMREAD_COLOR)
imgSegOrig=cv.imread('C:/Users/walee/Documents/DipCode/HaseebAssignment/lesion/IMD002_lesion.bmp',cv.IMREAD_GRAYSCALE)  
regObj=regionExtraction(imgSegOrig)
contours=regObj.differentiation()

imgOut=regObj.segmentOut(imgCol,contours)
cv.imshow('img',imgOut)
cv.waitKey()
cv.destroyAllWindows()