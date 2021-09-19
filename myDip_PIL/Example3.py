from PIL import Image
from pylab import *
img= array(Image.open('C:/Users/walee/Desktop/myDip/SampleImages/Samp4.jpg'))

x=[100,100,400,400]
y=[200,300,200,300]

plot(x,y,'r*')

plot(x[:2],y[:2],'r*-')
plot(x[2:],y[2:],'go-')
plot(x[1:4:2],y[1:4:2])
plot(x[0:3:2],y[0:3:2])
plot(x[1:3],y[1:3])
plot((x[0:4:3]),(y[0:4:3]))
title('Plotting Samp4.jpg')
show()

axis('off')