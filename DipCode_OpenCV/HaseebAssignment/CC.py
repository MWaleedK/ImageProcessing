import threading
import numpy as np



class ConnectedComponents:
    def __init__(self, img, totalLabels=[0], check=False, labelCount=0, outputImg=np.array(0),roCount=1,colCount=1,mutex=1):
        self.img = img
        self.roCount=roCount
        self.check = check
        self.rows=img.shape[0]
        self.columns=img.shape[1]
        self.labelCount = labelCount
        self.totalLabels = totalLabels
        self.mutex = threading.Lock()
        self.outputImg = np.zeros(self.img.shape, dtype=np.uint8)
        self.colCount=colCount

    def _4Connectivity(self):
        img2 = np.zeros(self.img.shape, np.uint8)
        labels = list()
        labelInc = 0
        labels.append(labelInc)
        for i in range(1, self.rows, 1):
            for j in range(1, self.columns, 1):
                if self.img[i][j] == 255:
                    if (self.img[i][j - 1] == 0 and self.img[i - 1][j] == 0):
                        labelInc = labelInc + 1
                        labels.append(labelInc)
                        img2[i][j]=labelInc
                    elif self.img[i - 1][j] == 255:
                        img2[i][j] = img2[i - 1][j]

                    elif self.img[i][j - 1] == 255:
                        img2[i][j] = img2[i][j - 1]

                    if (self.img[i][j - 1] == 255 and self.img[i - 1][j] == 255):
                        img2[i][j] == min(img2[i][j - 1], img2[i - 1][j])
                        labels[max(img2[i][j - 1], img2[i - 1][j])] = img2[i][j]
        return img2

    def secondPass_for_8(self):
        for i in range(1, self.rows, 1):
            for j in range(1, self.columns, 1):
                self.outputImg[i][j] = self.totalLabels[self.outputImg[i][j]]



    def connectedComp_8(self):

        t = threading.Thread(target=self.secondPass_for_8)
        for i in range(1, self.rows-2, 1):
            for j in range(1, self.columns-2, 1):
                if (self.img[i][j] == 255):
                    if (self.img[i][j - 1] == 0 and self.img[i - 1][j - 1] == 0 and self.img[i - 1][j] == 0 and
                            self.img[i - 1][j + 1] == 0):
                        self.labelCount = self.labelCount + 1
                        self.totalLabels.append(self.labelCount)
                        self.outputImg[i][j] = self.labelCount
                    elif (self.img[i][j - 1] == 255):
                        self.outputImg[i][j] = self.outputImg[i][j - 1]
                    elif (self.img[i - 1][j - 1] == 255):
                        self.outputImg[i][j] = self.outputImg[i - 1][j - 1]
                    elif (self.img[i - 1][j] == 255):
                        self.outputImg[i][j] = self.outputImg[i - 1][j]
                    elif (self.img[i - 1][j + 1] == 255):
                        self.outputImg[i][j] = self.outputImg[i - 1][j + 1]

                    if (self.img[i][j - 1] == 255 and self.img[i - 1][j - 1] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i][j - 1], self.outputImg[i - 1][j - 1])
                        self.totalLabels[max(self.outputImg[i][j - 1], self.outputImg[i - 1][j - 1])] = self.img[i][j]
                    if (self.img[i][j - 1] == 255 and self.img[i - 1][j] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i][j - 1], self.outputImg[i - 1][j])
                        self.totalLabels[max(self.outputImg[i][j - 1], self.outputImg[i - 1][j])] = self.img[i][j]
                    if (self.img[i][j - 1] == 255 and self.img[i - 1][j + 1] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i][j - 1], self.outputImg[i - 1][j + 1])
                        self.totalLabels[max(self.outputImg[i][j - 1], self.outputImg[i - 1][j + 1])] = self.img[i][j]
                    if (self.img[i - 1][j - 1] == 255 and self.img[i - 1][j] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i - 1][j - 1], self.outputImg[i - 1][j])
                        self.totalLabels[max(self.outputImg[i - 1][j - 1], self.outputImg[i - 1][j])] = \
                        self.outputImg[i][j]
                    if (self.img[i - 1][j - 1] == 255 and self.img[i - 1][j + 1] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i - 1][j - 1], self.outputImg[i - 1][j + 1])
                        self.totalLabels[max(self.outputImg[i - 1][j - 1], self.outputImg[i - 1][j + 1])] = \
                        self.outputImg[i][j]
                    if (self.img[i - 1][j] == 255 and self.img[i - 1][j + 1] == 255):
                        self.outputImg[i][j] = min(self.outputImg[i - 1][j], self.outputImg[i - 1][j + 1])
                        self.totalLabels[max(self.outputImg[i - 1][j], self.outputImg[i - 1][j + 1])] = \
                        self.outputImg[i][j]



        if (self.check == False):
            t.start()
        t.join()

        return self.outputImg.astype(np.uint8)

