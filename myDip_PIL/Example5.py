from pylab import *
from PIL import Image
from numpy import *

img=array(Image.open('C:/Users/walee/Desktop/myDip/SampleImages/Samp4.jpg'))
print (img.shape, img.dtype)
imshow(img)
print('Click 3 times')
x= ginput(3)
print('you clicked:',x)
show()
newIm=Image.fromarray(uint8(img))
print('fromarray')
(newIm.convert('L')).show()