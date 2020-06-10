import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import random
import cv2

def scaling_tech(x,method = "normalization"):
    if method == "normalization":
        x = (x-x.min())/(x.max()-x.min()+0.0001)
    else:
        x = (x-np.std(x))/x.mean()
    return x

def load_data(dir,width,height,shuffle=True, split_ratio = 0.8):
    # Step 1: Load Data...............
    subdirs = os.listdir(dir)
    subdirs = sorted(subdirs)

    labels = []
    images = []
    img_ext = [".jpg",".png",".jpeg"]

    n = len(subdirs)
    for i in range(n):
        filenames = os.listdir(dir+subdirs[i])
        print("class",subdirs[i],"contains",len(filenames),"images")
        for j in range(len(filenames)):
            ext = (os.path.splitext(filenames[j]))[1]
            if ext in img_ext:
                # img_filename = dir + subdirs[i]+"/" + filenames[j]
                img_filename = os.path.join(dir, subdirs[i], filenames[j])
                img   = cv2.imread(img_filename)
                image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA) # cv2.INTER_CUBIC
                # data_augmentation
                img_v = image[:,::-1]
                # img_h = image[::-1,:]
                # img_h_v = image[::-1,::-1]
                images.append(image)
                images.append(img_v)
                labels.append(i)
                labels.append(i)

    images = np.array(images)
    labels = np.array(labels)

    # Step 2: Normalize Data..........
    images = scaling_tech(images,method="normalization")

    # Step 4: Shuffle Data............
    if shuffle == True:
        indics = np.arange(0,len(images))
        np.random.shuffle(indics)

        labels = labels[indics]
        images = images[indics]

    # Step 5: Split Data.............
    # images = images.reshape((-1,28*28*3))
    m = int(len(images)*split_ratio)
    train_X = images[:m]
    train_Y = labels[:m]
    valid_X = images[m:]
    valid_Y = labels[m:]
    return train_X,train_Y,valid_X,valid_Y
