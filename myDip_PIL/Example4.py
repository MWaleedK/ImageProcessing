from PIL import Image
from pylab import *

img=array(Image.open('C:/Users/walee/Desktop/myDip/SampleImages/Samp1.jpg').convert('L'))

figure()
gray()
contour(img,origin='image')
axis('equal')
axis('off')
figure()
hist(img.flatten(),128)
show()