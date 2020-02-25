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
num='02'
# no index: Original
# index 00: Affine, Snowflakes
# index 01: Multiply, Perspective, Picewise, Elastic, Affine
# index 02: Affine, CoarseDropout(p=0.1, size_percent=0.15 )
# index 03: more than 4 person in 1 pic
def convert_bbs_from_yolo(xc,yc,w,h,label):
    # yolo format: x_center, y_center, width, height
    # imgaug bbs format: x_min, y_min, x_max, y_max
    return int(xc-w/2), int(yc-h/2), int(xc+w/2), int(yc+h/2),label

def convert_bbs_back_to_yolo(x1,y1,x2,y2,label):
    # imgaug bbs format: x_min, y_min, x_max, y_max
    # yolo format: x_center, y_center, width, height
    return int(label), int((x1+x2)/2),int((y1+y2)/2),int(x2-x1),int(y2-y1)


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

        H = img.shape[0]
        W = img.shape[1]
        
        bbs = []
        
        for box in boxes:
            x1, y1, x2, y2, l = convert_bbs_from_yolo(box[1]*W, box[2]*H, box[3]*W, box[4]*H, box[0])
            bbs.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,label=l))

        bbs=ia.BoundingBoxesOnImage(bbs, img.shape)

        for repeat in range(10):
            ia.seed(np.random.randint(0,1000))
            np.random.seed(np.random.randint(0,1000))
            seq = iaa.Sequential([
                    # iaa.Multiply((1.1, 1.1)),
                    # iaa.Fog(),
                    # iaa.PerspectiveTransform(scale=0.05),
                    # iaa.PiecewiseAffine(scale=0.015),
                    # iaa.ElasticTransformation(sigma=3, alpha = 3),
                    # iaa.Superpixels(p_replace = 0.1, n_segments=100),
                    # iaa.pillike.EnhanceSharpness(factor=5),
                    iaa.Affine(
                        translate_px={"x": int(np.random.normal(0,10)), "y": int(np.random.normal(0,10))},
                        scale=(np.random.normal(1,0.15),np.random.normal(1,0.15)),
                        rotate=np.random.normal(loc=0,scale=5),
                        mode='edge'
                    ),
                    # iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),
                    iaa.CoarseDropout(p=np.random.uniform(0.02,0.1), size_percent=np.random.uniform(0.1,0.8) )
                ])
            image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
            
            bbs_yolo_save = []
            for i in range(len(bbs.bounding_boxes)):
                before = bbs.bounding_boxes[i]
                after = bbs_aug.bounding_boxes[i]
                # print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                #     i,
                #     before.x1, before.y1, before.x2, before.y2,
                #     after.x1, after.y1, after.x2, after.y2)
                # )
                l,x,y,w,h = convert_bbs_back_to_yolo(after.x1, after.y1, after.x2, after.y2, after.label)
                bbs_yolo_save.append([int(l),x/W,y/H,w/W,h/H])

            image_before = bbs.draw_on_image(img)
            image_after = bbs_aug.draw_on_image(image_aug, color=[0, 0, 255])
            name = os.path.splitext(base)[0]+'_'+num+'_'+str(repeat)
            # name = os.path.splitext(base)[0]
            directory = 'E:/KVOD/Augmentation/img_aug/'
            imsave(directory+name+'.jpg',image_aug)
            np.savetxt(directory+name+'.txt',bbs_yolo_save, fmt=['%d', '%0.6f','%0.6f','%0.6f','%0.6f'])
            
            # ia.imshow(image_before)
            # ia.imshow(image_after)