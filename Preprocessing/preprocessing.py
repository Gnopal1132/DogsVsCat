import numpy as np
import cv2
import os
from pathlib import Path
import albumentations as A


def read_image(path, size, to_aug=False):
    path = Path(path)
    if not path.exists():
        raise Exception("Image Not Found")
    else:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # By default cv2 reads in Blue,Green,Red Format to convert in RGB
        image = cv2.resize(image, size)
        if to_aug:
            image = Augment_me(image)
        image = scaled_img(image)
        label = 0 if 'cat' in str(os.path.basename(path)) else 1
        return image, label


def read_image_test(path, size):
    path = Path(path)
    if not path.exists():
        raise Exception("Image Not Found")
    else:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # By default cv2 reads in Blue,Green,Red Format to convert in RGB
        image = cv2.resize(image, size)
        image = scaled_img(image)
        return np.asarray(image)


def scaled_img(image):
    image = np.asarray(image)
    mean = np.mean(image, axis=0, keepdims=True)
    return (image - mean)


def Augment_me(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomRotate90(p=0.5)
    ])
    return transform(image=image)["image"]