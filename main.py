import os
from os import walk
import PIL
from PIL import Image
import subprocess
import sys
import datetime
import time

os.system
start = time.time()

img_dir = "images/"
out_dir = "output/"

text_file = open('imgpath.txt', 'w')

#List of Images Path
txt_img_list = []

#Iteration for showing pictures that we want to process
for (dirpath, dirnames, filenames) in walk(img_dir):
    txt_img_list.extend(filenames)
    break
print(txt_img_list)
text_file.write("\n")
#Process
for txt_name in txt_img_list:
    #Save images path
    text_file.write('images/%s.jpg\n'%(os.path.splitext(txt_name)[0]))

text_file.close()

#Variable Init
d = 1
total = 0

"""Reading image path"""
with open('imgpath.txt', 'r') as fobj:
    for line in fobj:
        image_List = [[num for num in line.split()] for line in fobj]

open("larvaNumber.txt", "w").close()

#Iteration for detecting images
for images in image_List:
    commands = ["darknet.exe", "detector", "test", "data/obj.data", "cfg/small-larva-test.cfg", "newweight/small-larva_2000.weights" ,"-dont_show", images[0]]
    os.system(', '.join(commands))

    output = subprocess.check_output(commands)
    output = output.decode("utf-8").split("\n")

    #Count the number of lines that contain "larva
    numLarva = len([i.split(":")[0] for i in output if i.split(":")[0] == 'larva'])

    print("Ada {} larva yang terdeteksi.".format(numLarva))

    #Write total of larva in the frame into larvaNumber.txt
    with open("larvaNumber.txt", "a") as myfile:
        myfile.write("Pada gambar ke {} Ada {} Larva yang terdeteksi\n".format(d, numLarva))


    #Save predictions image into output folder
    predicted_image = Image.open("predictions.jpg")
    output_image = "output/predicted_image%3d.jpg"%d
    predicted_image.save(output_image)

    d+=1
    end = time.time()
    print("WAKTU PROSES adalah ",end - start, "detik")
    total=total+numLarva

end = time.time()
print("TOTAL WAKTU PROSES adalah ",end - start, "detik")
print("TOTAL ADA ", total, "LARVA")
