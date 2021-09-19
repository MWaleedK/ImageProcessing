import cv2 as cv
from PIL import Image
import numpy as np
from scipy import stats
import glob
from os import read
from numpy import sqrt
import csv


def errorCalc(x1,y1,x2,y2):
    dists=list()
    dists.append('distances')
    for i in range(0,len(x2),1):
        dists.append(sqrt(((x2[i]-x1[i])**2)+((y2[i]-y1[i])**2)))
    return dists

def checkingVals(path):
    x1=list()
    y1=list()
    items=list()
    with open(path) as ODC:
        reader=csv.DictReader(ODC)
        items=[item for item in reader]
    x1=[float(item['x']) for item in items]
    y1=[float(item['y']) for item in items]
    return (x1,y1)

def OutputCreator(input_file, output_file, appCol):
   
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        csv_reader = csv.reader(read_obj)
        csv_writer = csv.writer(write_obj)
        count=0
        for row in csv_reader:
            row.append(appCol[count])
            count=count+1
            csv_writer.writerow(row)



def oneForAll(path):
    testFiles = glob.glob(path)
    x2=list()
    y2=list()
    index1=0
    for image in testFiles:
        img = cv.imread(image, cv.IMREAD_COLOR)
        x,y=(diskCoordinates(img,index1))
        x2.append(x)
        y2.append(y)
        index1=index1+1
    return (x2,y2)

def diskCoordinates(img,index1):
    index2=0

    #img_33.show('img')
    cv.imwrite(f'{imwritePath}/___{[index1]}{[index2]}.png',img.astype(np.uint8))
    index2=index2+1
    
    
    
    #picture grayed
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #thresholded
    res, arr = cv.threshold(gray,percentileVals[index1],255,cv.THRESH_BINARY)
    
    
    #img_34.show('THRESH')
    cv.imwrite(f'{imwritePath}/___{[index1]}{[index2]}.png',arr.astype(np.uint8))
    index2=index2+1
    
    #CCA
    labels, ret = cv.connectedComponents(arr, connectivity=8)
    
    
    
    #img_35.show('CCA')
    cv.imwrite(f'{imwritePath}/___{[index1]}{[index2]}.png',ret.astype(np.uint8))
    index2=index2+1
    
    #highest frequency label detection
    mode, count= stats.mode(ret[ret>1], axis=None)
    

    #Only keeping the highest frequency label
    row,col = ret.shape
    for i in range(0, row):
        for j in range(0, col):
            if ret[i][j]==mode:
                ret[i][j] = 255
            else:
                ret[i][j] = 0

    
    #img_36.show('ret')
    cv.imwrite(f'{imwritePath}/___{[index1]}{[index2]}.png',ret.astype(np.uint8))
    index2=index2+1
    

    #midpoint calculation
    sx=0
    fx=0
    sy=0
    fy=0
    c=False
    for i in range(0, row):
        for j in range(0, col):
            if ret[i][j]==255:
                if c == False:
                    sx= i
                    sy= j
                    c = True
                elif fx < i:
                    fx = i
                elif fy < j:
                    fy = j

    #print(sx,fx, sy, fy)
    cx=(fx-sx)/2
    cy=(fy-sy)/2
    #print(cx,cy)

    x=int(sx+cx)
    y=int(sy+cy)
    print(x,y)


    #auto arrow drawing
    l=int((row+col)/100)
    w=int(l/5)
    print(l,w)

    for i in range(x-w, x+w):
        for j in range(y-l, y+l):
            img[i][j]=0
    for j in range(y-w, y+w):
        for i in range(x-l, x+l):
            img[i][j]=0

    #output
    
    #img_33.show()
    cv.imwrite(f'{imwritePath}/___{[index1]}{[index2]}.png',img.astype(np.uint8))
    index2=index2+1
    

    return x,y# returning x2,y2



imageDirPath='C:/Users/walee/Documents/DipCode/Assignments/Specimens/Ass_2/FundusImage/*.*'
pathCSVinp='C:/Users/walee/Documents/DipCode/Assignments/optic_disc_centres.csv'
pathCSVout='C:/Users/walee/Documents/DipCode/Assignments/output.csv'
imwritePath='C:/Users/walee/Documents/DipCode/Assignments/Specimens/Ass_2/MarkedFundus'
percentileVals=[147,170,170,170,170,48,128,170,170,128,170,170,170,170,170,170,170,170,128,128,128,170,170,128,128,170,128,170,170,170,128,170,170,128,150,128,150,150,84,128,128,128,128,150,150,128,150,150,78,150,150,150,138,114,150,150,150,150,150,150,150,150]

print(percentileVals)
[x1,y1]=checkingVals(pathCSVinp)
[x2,y2]=oneForAll(imageDirPath)
diff=errorCalc(x1,y1,x2,y2)
OutputCreator(pathCSVinp,pathCSVout,diff)

cv.waitKey()
cv.destroyAllWindows()