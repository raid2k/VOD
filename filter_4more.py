import imgaug.augmenters as iaa
import imgaug as ia
from matplotlib.image import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
img_dir = 'C:/Users/gnt/Desktop/Codes/data/SOC_Center/08/original/190827/passenger_5_suv/095519_0/RIGHT_190827_095519.jpg'
label_dir = 'C:/Users/gnt/Desktop/Codes/data/Annotations/*.txt'
label_list = glob.glob(label_dir)

for label in label_list:
    base = os.path.basename(label)
    if base == 'classes.txt':
        continue
    try:
        img_dir = 'C:/Users/gnt/Desktop/Codes/data/img/'+os.path.splitext(base)[0]+'.jpg'
        img = imread(img_dir)
    except:
        print(base)
        continue

    if base[0] == 'L' or base[0] == 'R': 
        # print(base)

        boxes = np.loadtxt(label, ndmin=2)
        people_cnt = 0
        for box in boxes:
            if box[0] == 3 or box[0] == 4:
                people_cnt += 1
        if people_cnt >=4:
            name = os.path.splitext(base)[0]
            
            imsave('C:/Users/gnt/Desktop/Codes/data/img_4_more/'+name+'.jpg',img)
            np.savetxt('C:/Users/gnt/Desktop/Codes/data/img_4_more/'+name+'.txt',boxes, fmt=['%d', '%0.6f','%0.6f','%0.6f','%0.6f'])

            