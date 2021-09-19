import cv2 as cv
from PIL import Image
import numpy as np
from scipy import stats

#picture read color
img=cv.imread('C:\zdata\Study\Sem 6\DIP\Assignments\Assgmt 2\Fundus image/1.JPG', cv.IMREAD_COLOR)
#val=img[[541],[1530]]
#print(val)
img_33=Image.fromarray(img)
img_33.show()

#picture grayed
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#thresholded
res, arr = cv.threshold(gray,127,255,cv.THRESH_BINARY)
img_34=Image.fromarray(arr)
img_34.show('THRESH')
#val=arr[[541],[1530]]
#print(val)

#CCA
labels, ret = cv.connectedComponents(arr, connectivity=8)
#print(labels)
img_35=Image.fromarray(ret)
img_35.show('CCA')

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

img_36=Image.fromarray(ret)
img_36.show()


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