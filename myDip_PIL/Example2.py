
# importing Image class from PIL package  
from PIL import Image 
   
# creating a object  
image = Image.open(r"C:/Users/walee/Desktop/myDip/SampleImages/Samp3.png").convert('L')
MAX_SIZE = (500, 500) 
box=[100,100,400,400]#dimensions... left top right bottom
image=image.crop(box)#crop
image=image.resize((1280,720))#resize
image=image.rotate(90)#rotate
#image.thumbnail(MAX_SIZE) 
  
# creating thumbnail 
image.show() 