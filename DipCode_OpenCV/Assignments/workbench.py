import cv2 as cv
from PIL import Image
import numpy as np
from scipy import stats


fil5x5=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
fil7x7=np.array([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]])
fil9x9=np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]])
filLaplac3x3=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

def AvgFilter(arr,v):
    rows=arr.shape[0]
    columns=arr.shape[1]
    avgValue=1/(rows*columns)
    return arr*avgValue

def averaging(img,filter):
    row=filter.shape[0]
    column=filter.shape[1]
    img2=np.lib.pad(img,((int(column/2),int(row/2)),(int(column/2),int(row/2))),mode='constant',constant_values=(0,0))
    cv.normalize(img2,img2,0,255,cv.NORM_MINMAX)
    row_=int(0)
    while(row_<img.shape[1]):
        column_=int(0)
        while(column_<img.shape[1]):
            try:
                img2[row_:row+row_,column_:column+column_]=(img[row_:row+row_,column_:column+column_]*filter).sum()
            except:
                pass
            column_=column_+1
        row_=row_+1
    return img2.astype(np.uint8)


def preprocessing(img,choice,rows,columns):
    if choice=='edge':
        img=np.lib.pad(img,((int(rows/2),int(columns/2)),(int(rows/2),int(columns/2))),'edge')
    else:
        img=np.lib.pad(img,((int(rows/2),int(columns/2)),(int(rows/2),int(columns/2))),'constant', constant_values=(0,0))
    return img.astype(np.uint8)

def minMaxAvgFilter(img,rows,cols,filtype='median'):
    print(filtype)
    img2=preprocessing(img,'edges',rows,cols)
    row_=int(0)
    while(row_<img2.shape[1]):
        column_=int(0)
        while(column_<img2.shape[1]):
            try:
                if(filtype=='min'):
                    img2[row_:rows+row_,column_:cols+column_]=np.amin(img[row_:rows+row_,column_:cols+column_])
                elif(filtype=='max'):
                    img2[row_:rows+row_,column_:cols+column_]=np.amax(img[row_:rows+row_,column_:cols+column_])
                else:
                    img2[row_:rows+row_,column_:cols+column_]=np.nanmedian(img[row_:rows+row_,column_:cols+column_])
            except:
                pass
            column_=column_+1
        row_=row_+1
    return img2.astype(np.uint8)

#picture read color
#img=cv.imread('C:/Users/walee/Documents/DipCode/Assignments/Specimens/Ass_2/FundusImage/1ffa92e4-8d87-11e8-9daf-6045cb817f5b..JPG', cv.IMREAD_COLOR)
img=cv.imread('C:/Users/walee/Documents/DipCode/Assignments/Specimens/Ass_2/FundusImage/1ffa9555-8d87-11e8-9daf-6045cb817f5b..JPG',cv.IMREAD_COLOR)
#val=img[[541],[1530]]
#print(val)
img_33=Image.fromarray(img)
img_33.show()

#picture grayed
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#thresholded
res, arr = cv.threshold(gray,127,255,cv.THRESH_BINARY)
#img_34=Image.fromarray(arr)
#img_34.show('THRESH')
#val=arr[[541],[1530]]
#print(val)


print('Here IrnBru')
aidas=minMaxAvgFilter(arr,7,7,'median')
img65=Image.fromarray(arr.astype(np.uint8))
img65.show()

#CCA
labels, ret = cv.connectedComponents(arr, connectivity=8)
#print(labels)
#img_35=Image.fromarray(ret)
#img_35.show('CCA')

#highest frequency label detection
mode, count= stats.mode(ret[ret>1], axis=None)
#val=ret[[541],[1530]]
#print(val)


#Only keeping the highest frequency label
row,col = ret.shape
for i in range(0, row):
    for j in range(0, col):
        if ret[i][j]==mode:
            ret[i][j] = 255
        else:
            ret[i][j] = 0

#img_36=Image.fromarray(ret)
#img_36.show('HighFreqLabel')


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
dst = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_33=Image.fromarray(dst)
img_33.show()

cv.waitKey()
cv.destroyAllWindows()