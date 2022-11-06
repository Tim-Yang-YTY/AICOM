import logging
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
import os
import random
from skimage.util import random_noise
from numpy import expand_dims
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import torchvision.transforms as T

import torch
from PIL import Image, ImageOps

def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

def add_s_p_noise_testing(img):
    row, col, _ = img.shape
    number_of_pixels = random.randint(1000, 5000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(1000, 5000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img

def create_testing_dataset(classpath,working_dir='./'):
    flist = [f for f in os.listdir(classpath) if not f.startswith('.')]

    print(f'original length: {len(flist)}')
    aug_dir=os.path.join(working_dir, 'aug_testing_dataset_2/Others/')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)

    '(1) Random rotation by a multiple of 90 degrees, ' \
    '(2) Random rotation in the range of-45 to 45 degrees,' \
    ' (3) Translation, (4) Reflection, (5) Shear, (6) Hue jitter, (7) Saturation jitter, (8) Brightness jitter, ' \
    '(9) Contrast jitter, (10) Salt and Pepper Noise, (11) Gaussian Noise, (12) Synthetic Blur and (13) Scaling.'

    for image in flist:
        print(f'########################Augmenting {image}##########################')

        image = os.path.join(classpath, image)

        img = Image.open(image).convert("RGB")
        img_name = os.path.splitext(os.path.basename(image))[0]
        gen = ImageDataGenerator(vertical_flip=True, rotation_range=25, zoom_range=[1.3, 1.7],
                                 brightness_range=[0.3, 1.7],width_shift_range=.4,height_shift_range=.4, fill_mode = 'constant', channel_shift_range=10)

        aug_gen = gen.flow(expand_dims(img, 0), batch_size=1, shuffle=False,
                                          save_to_dir=aug_dir, save_prefix='aug-testing',
                                          save_format='jpg')
        aug_img_count = 0
        delta = 7
        while aug_img_count < delta:
            images = next(aug_gen)

            aug_img_count += len(images)
        # exit()
        ################################################################################################################################################
        angle = [70,140,210,280]
        rotated_img = img.rotate(random.choice(angle), expand=True)
        rotated_img=rotated_img.convert('RGB')
        rotated_img.save(
            aug_dir+'/' + img_name + '-testing-01.jpg')
        ################################################################################################################################################
        transform = T.RandomRotation((-170,170))
        rotated_img = transform(img)
        rotated_img=rotated_img.convert('RGB')
        rotated_img.save(
            aug_dir+'/' + img_name + '-testing-02.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=170,translate=(0.5,0.4))
        translated_img = transform(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-03.jpg')
        ################################################################################################################################################
        im_mirror = ImageOps.flip(img)
        im_mirror=im_mirror.convert('RGB')
        im_mirror.save(aug_dir+'/' + img_name + '-testing-04.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=170,shear=0.8)
        translated_img = transform(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-05.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(hue=.5)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-06.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(saturation=.7)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-07.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(brightness=.8)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-08.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(contrast=.8)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-testing-09.jpg')
        ################################################################################################################################################

        cv2.imwrite(aug_dir+'/' + img_name + '-testing-10.jpg',
                    add_s_p_noise_testing(cv2.imread(image)))


        orig_img = Image.open(image)
        blurred_img = T.GaussianBlur(kernel_size=(51, 91), sigma=4)(orig_img).convert('RGB')

        noise_img = add_noise(T.ToTensor()(orig_img), 0.2)
        noise_img = T.ToPILImage()(noise_img).convert('RGB')

        img_name = os.path.splitext(os.path.basename(image))[0]

        blurred_img.save(
            aug_dir+'/' + img_name + '-testing-12.jpg')
        noise_img.save(
            aug_dir+'/' + img_name + '-testing-11.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=70,scale=(1.6,1.8))
        scaled_img = transform(img)
        scaled_img=scaled_img.convert('RGB')
        scaled_img.save(
            aug_dir+'/' + img_name + '-testing-13.jpg')
    print("#############################################################Finished#############################################################")

def main():
    create_testing_dataset('Original_Images/Original_Images/Others')


if __name__ == "__main__":
    main()
