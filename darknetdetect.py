import os
import glob
import time
os.system
import PIL
import PIL.Image as Image

d = 1
#path to txt file
test_Path = r'D:\darknet\build\darknet\x64\scripts\test'
with open((test_Path + '.txt'),'r') as fobj:
    for line in fobj:
        image_List = [[num for num in line.split()] for line in fobj]
for images in image_List:
    commands = ['darknet.exe detector test data/obj.data cfg/yolov3-clone.cfg yolov3-clone_final.weights -dont_show', images[0]]
    os.system(' '.join(commands))

    predicted_image = Image.open("predictions.jpg")
    output = "D:/darknet/build/darknet/x64/scripts/predicted_image%2d.jpg"%d
    predicted_image.save(output)
    d+=1
