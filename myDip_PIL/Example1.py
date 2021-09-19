from PIL import Image
import os
from imtools import get_imlist

filelist= get_imlist('C:/Users/walee/Desktop/myDip/SampleImages')

for infile in filelist:
    outfile=os.path.splitext(infile)[0] + '.jpg'
    if infile != outfile:
        try:
            im_image=Image.open(infile).save(outfile)
        except IOError:
            print("cannot convert",infile)


        
