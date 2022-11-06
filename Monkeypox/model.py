import logging
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
import os
import random
from skimage.util import random_noise
from numpy import expand_dims
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import time
import torchvision.transforms as T
import torch
from PIL import Image, ImageOps



def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy
def add_blur_noise_train(df,noise_factor=0.3,sigma=3):
    aug_dir = os.path.join(r'./', 'noise_blur_aug_train3')  # directory to store augmented images

    if os.path.isdir(aug_dir):  # start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)  # make class directories within aug directory
    # create and store the augmented images
    total = 0

    groups = df.groupby('labels')  # group by class

    for label in df['labels'].unique():  # for every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
    #      # determine how many samples there are in this class
    #     # print(sample_count)
    #     # exit()
    #
        target_dir = os.path.join(aug_dir, label)  # define where to write the images
    #
    #     ############################################################################
    #
        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        image_names = group['filepaths']

        for image in image_names:
            # Change_Value(image, image_path, value_path)
            # Change_Saturation(image, image_path, sat_path)
            # break
            # print()
            orig_img = Image.open(image)
            blurred_img = T.GaussianBlur(kernel_size=(51,91), sigma=3)(orig_img)

            noise_img = add_noise(T.ToTensor()(orig_img), 0.1)
            noise_img = T.ToPILImage()(noise_img)


            img_name = os.path.splitext(os.path.basename(image))[0]

            # print(save_path)
            # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
            blurred_img.save(
                save_path + '/' + img_name + 'blured_sigma3.jpg')
            noise_img.save(
                save_path + '/' + img_name + 'noise_0.1.jpg')

    aug_fpaths = []
    aug_labels = []

    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))


    return df


def add_blur_noise_test(df,noise_factor=0.3,sigma=3):
    aug_dir = os.path.join(r'./', 'noise_blur_aug_test3')  # directory to store augmented images
    if os.path.isdir(aug_dir):  # start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)  # make class directories within aug directory
    # create and store the augmented images
    total = 0

    groups = df.groupby('labels')  # group by class

    for label in df['labels'].unique():  # for every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
    #      # determine how many samples there are in this class
    #     # print(sample_count)
    #     # exit()
    #
        target_dir = os.path.join(aug_dir, label)  # define where to write the images
    #
    #     ############################################################################
    #
        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        image_names = group['filepaths']
        dH = 30
        dS = 0.2
        dV = 0.2
        # print(image_names)
        for image in image_names:
            # Change_Value(image, image_path, value_path)
            # Change_Saturation(image, image_path, sat_path)
            # break
            # print()
            orig_img = Image.open(image)
            blurred_img = T.GaussianBlur(kernel_size=(51,91), sigma=3)(orig_img)

            noise_img = add_noise(T.ToTensor()(orig_img), 0.1)
            noise_img = T.ToPILImage()(noise_img)


            img_name = os.path.splitext(os.path.basename(image))[0]

            # print(save_path)
            # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
            blurred_img.save(
                save_path + '/' + img_name + 'blured_sigma3.jpg')
            noise_img.save(
                save_path + '/' + img_name + 'noise_0.1.jpg')

    aug_fpaths = []
    aug_labels = []

    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))

    return df

def add_blur_noise_val(df,noise_factor=0.3,sigma=3):
    aug_dir = os.path.join(r'./', 'noise_blur_aug_val3')  # directory to store augmented images
    if os.path.isdir(aug_dir):  # start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)  # make class directories within aug directory
    # create and store the augmented images
    total = 0

    groups = df.groupby('labels')  # group by class

    for label in df['labels'].unique():  # for every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
         # determine how many samples there are in this class
        # print(sample_count)
        # exit()

        target_dir = os.path.join(aug_dir, label)  # define where to write the images
    #
    #     ############################################################################
    #
        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        image_names = group['filepaths']
        dH = 30
        dS = 0.2
        dV = 0.2
        # print(image_names)
        for image in image_names:
            # Change_Value(image, image_path, value_path)
            # Change_Saturation(image, image_path, sat_path)
            # break
            # print()
            orig_img = Image.open(image)
            blurred_img = T.GaussianBlur(kernel_size=(51,91), sigma=3)(orig_img)

            noise_img = add_noise(T.ToTensor()(orig_img), 0.1)
            noise_img = T.ToPILImage()(noise_img)


            img_name = os.path.splitext(os.path.basename(image))[0]

            # print(save_path)
            # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
            blurred_img.save(
                save_path + '/' + img_name + 'blured_sigma3.jpg')
            noise_img.save(
                save_path + '/' + img_name + 'noise_0.1.jpg')

    aug_fpaths = []
    aug_labels = []

    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))

    return df







class LR_ASK(keras.callbacks.Callback):
    def __init__(self, model, epochs, ask_epoch):  # initialization of the callback
        super(LR_ASK, self).__init__()
        self.model = model
        self.ask_epoch = ask_epoch
        self.epochs = epochs
        self.ask = False  # if True query the user on a specified epoch
        self.lowest_vloss = np.inf
        self.best_weights = self.model.get_weights()  # set best weights to model's initial weights
        self.best_epoch = 1


    def on_train_begin(self, logs=None):  # this runs on the beginning of training
        if self.ask_epoch == 0:
            print('you set ask_epoch = 0, ask_epoch will be set to 1', flush=True)
            self.ask_epoch = 1
        if self.ask_epoch >= self.epochs:  # you are running for epochs but ask_epoch>epochs
            print('ask_epoch >= epochs, will train for ', self.epochs, ' epochs', flush=True)
            self.ask = False  # do not query the user
        if self.epochs == 1:
            self.ask = False  # running only for 1 epoch so do not query user
        else:
            print('Training will proceed until epoch', self.ask_epoch, ' then you will be asked to')
            print(' enter H to halt training or enter an integer for how many more epochs to run then be asked again')
        self.start_time = time.time()  # set the time at which training started

    def on_train_end(self, logs=None):  # runs at the end of training
        print('loading model with weights from epoch ', self.best_epoch)
        self.model.set_weights(self.best_weights)  # set the weights of the model to the best weights
        tr_duration = time.time() - self.start_time  # determine how long the training cycle lasted
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg, flush=True)  # print out training duration time

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch

        v_loss = logs.get('val_loss')  # get the validation loss for this epoch
        if v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model.get_weights()  # set best weights to model's initial weights
            self.best_epoch = epoch + 1
            print(
                f'\n validation loss of {v_loss:7.4f} is below lowest loss, saving weights from epoch {str(epoch + 1):3s} as best weights')
        else:
            print(
                f'\n validation loss of {v_loss:7.4f} is above lowest loss of {self.lowest_vloss:7.4f} keeping weights from epoch {str(self.best_epoch)} as best weights')

        if self.ask:  # are the conditions right to query the user?
            if epoch + 1 == self.ask_epoch:  # is this epoch the one for quering the user?
                print(
                    '\n Enter H to end training or  an integer for the number of additional epochs to run then ask again')
                ans = input()

                if ans == 'H' or ans == 'h' or ans == '0':  # quit training for these conditions
                    print('you entered ', ans, ' Training halted on epoch ', epoch + 1, ' due to user input\n',
                          flush=True)
                    self.model.stop_training = True  # halt training
                else:  # user wants to continue training
                    self.ask_epoch += int(ans)
                    if self.ask_epoch > self.epochs:
                        print('\nYou specified maximum epochs of as ', self.epochs, ' cannot train for ',
                              self.ask_epoch, flush=True)
                    else:
                        print('you entered ', ans, ' Training will continue to epoch ', self.ask_epoch, flush=True)
                        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
                        print(f'current LR is  {lr:7.5f}  hit enter to keep  this LR or enter a new LR')
                        ans = input(' ')
                        if ans == '':
                            print(f'keeping current LR of {lr:7.5f}')
                        else:
                            new_lr = float(ans)
                            tf.keras.backend.set_value(self.model.optimizer.lr,
                                                       new_lr)  # set the learning rate in the optimizer
                            print(' changing LR to ', ans)

def balance_val(df, n, working_dir, img_size):
    df=df.copy()
    df = add_blur_noise_val(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir=os.path.join(working_dir, 'aug_val3')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total=0
    gen=ImageDataGenerator(rescale = 1./255, horizontal_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups=df.groupby('labels') # group by class

    for label in df['labels'].unique():  # for every class
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        if sample_count< 10*sample_count: # if the class has less than target number of images
            aug_img_count=0
            delta=10*sample_count - sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe(group,  x_col='filepaths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
        ############################################################################

        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        # image_names = group['filepaths']
        # dH = 30
        # dS = 0.2
        # dV = 0.2
        # # print(image_names)
        # for image in image_names:
        #     # Change_Value(image, image_path, value_path)
        #     # Change_Saturation(image, image_path, sat_path)
        #     # break
        #     # print()
        #     img = cv2.imread(image)
        #
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
        #     image_H, image_S, image_V = cv2.split(img_hsv)
        #     for h in range(3):
        #         change_H = image_H + dH * h
        #         change_H -= (change_H > 180) * 180
        #
        #         for s in range(4):
        #             change_S = image_S * (1 - s * dS)
        #             for v in range(3):
        #                 change_V = image_V * (1 - v * dV)
        #                 img_change = cv2.merge([change_H, change_S, change_V])
        #                 img_change = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        #                 # change_path = save_path + '/' + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)
        #                 # if not os.path.exists(change_path):
        #                 #     os.makedirs(change_path)
        #                 img_name = os.path.splitext(os.path.basename(image))[0]
        #
        #                 # print(save_path)
        #                 # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
        #                 cv2.imwrite(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)+'.jpg', img_change)

    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths=[]
    aug_labels=[]

    classlist=os.listdir(aug_dir)
    for klass in classlist:
        classpath=os.path.join(aug_dir, klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries=pd.Series(aug_fpaths, name='filepaths')
    Lseries=pd.Series(aug_labels, name='labels')
    aug_df=pd.concat([Fseries, Lseries], axis=1)
    df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df
def balance_test(df, n, working_dir, img_size):
    df=df.copy()
    df = add_blur_noise_test(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir=os.path.join(working_dir, 'aug_test3')# directory to store augmented images
    # aug_dir = '/Users/timothy/Downloads/aug_test'
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # # create and store the augmented images
    total=0
    gen=ImageDataGenerator(rescale = 1./255, horizontal_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups=df.groupby('labels') # group by class

    for label in df['labels'].unique():  # for every class
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        if sample_count< 10*sample_count: # if the class has less than target number of images
            aug_img_count=0
            delta=10*sample_count - sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe(group,  x_col='filepaths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
    #     ############################################################################
    #
        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        # image_names = group['filepaths']
        # dH = 30
        # dS = 0.2
        # dV = 0.2
        # # print(image_names)
        # for image in image_names:
        #     # Change_Value(image, image_path, value_path)
        #     # Change_Saturation(image, image_path, sat_path)
        #     # break
        #     # print()
        #     img = cv2.imread(image)
        #
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
        #     image_H, image_S, image_V = cv2.split(img_hsv)
        #     for h in range(3):
        #         change_H = image_H + dH * h
        #         change_H -= (change_H > 180) * 180
        #
        #         for s in range(4):
        #             change_S = image_S * (1 - s * dS)
        #             for v in range(3):
        #                 change_V = image_V * (1 - v * dV)
        #                 img_change = cv2.merge([change_H, change_S, change_V])
        #                 img_change = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        #                 # change_path = save_path + '/' + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)
        #                 # if not os.path.exists(change_path):
        #                 #     os.makedirs(change_path)
        #                 img_name = os.path.splitext(os.path.basename(image))[0]
        #
        #                 # print(save_path)
        #                 # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
        #                 cv2.imwrite(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)+'.jpg', img_change)

    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths=[]
    aug_labels=[]

    classlist=os.listdir(aug_dir)
    for klass in classlist:
        classpath=os.path.join(aug_dir, klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries=pd.Series(aug_fpaths, name='filepaths')
    Lseries=pd.Series(aug_labels, name='labels')
    aug_df=pd.concat([Fseries, Lseries], axis=1)
    df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df

def initial_balance_df(df, n, working_dir, img_size):
    df = df.copy()
    # df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'balanced_df')  # directory to store augmented images
    # if os.path.isdir(aug_dir):# start with an empty directory
    #     shutil.rmtree(aug_dir)
    # os.mkdir(aug_dir)
    # for label in df['labels'].unique():
    #     dir_path=os.path.join(aug_dir,label)
    #     os.mkdir(dir_path) # make class directories within aug directory
    # # create and store the augmented images
    # total = 0
    # gen = ImageDataGenerator(vertical_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2],
    #                          width_shift_range=.2,height_shift_range=.2, fill_mode = 'constant')
    # groups = df.groupby('labels')  # group by class
    # group1 = groups.get_group(df['labels'].unique()[0])
    # group2 = groups.get_group(df['labels'].unique()[1])
    # groups_len = [len(group1), len(group2)]
    # groups_diff = abs(len(group2) - len(group1))
    # if groups_diff != 0:
    #
    #     balance_len = min(groups_len)
    #     balance_len_index = groups_len.index(balance_len)
    #     print(len(group1), len(group2))
    #     print(balance_len, balance_len_index, groups_diff, df['labels'].unique()[balance_len_index])
    #
    #     group = groups.get_group(
    #         df['labels'].unique()[balance_len_index])  # a dataframe holding only rows with the specified label
    #     # determine how many samples there are in this class
    #     # print(sample_count)
    #     # exit()
    #     # if the class has less than target number of images
    #     aug_img_count = 0
    #     delta = groups_diff  # number of augmented images to create
    #
    #     target_dir = os.path.join(aug_dir, df['labels'].unique()[balance_len_index])  # define where to write the images
    #     print(target_dir)
    #     msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', df['labels'].unique()[
    #         balance_len_index], str(delta))
    #
    #     print(msg, '\r', end='')  # prints over on the same line
    #
    #     aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
    #                                       class_mode=None, batch_size=1, shuffle=False,
    #                                       save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
    #                                       save_format='jpg')
    #     while aug_img_count < delta:
    #
    #         images = next(aug_gen)
    #         aug_img_count += len(images)


    aug_fpaths = []
    aug_labels = []

    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))

    return df

def balance_df(df, n, working_dir, img_size):
    df=df.copy()
    df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir=os.path.join(working_dir, 'aug_balanced_df')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total=0
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                  height_shift_range=.2, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups=df.groupby('labels') # group by class

    for label in df['labels'].unique():  # for every class
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        if sample_count< 10*sample_count: # if the class has less than target number of images
            aug_img_count=0
            delta=10*sample_count - sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe(group,  x_col='filepaths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
        ############################################################################

        save_path = target_dir
        value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        # image_names = group['filepaths']
        # dH = 30
        # dS = 0.2
        # dV = 0.2
        # # print(image_names)
        # for image in image_names:
        #     # Change_Value(image, image_path, value_path)
        #     # Change_Saturation(image, image_path, sat_path)
        #     # break
        #     # print()
        #     img = cv2.imread(image)
        #
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
        #     image_H, image_S, image_V = cv2.split(img_hsv)
        #     for h in range(3):
        #         change_H = image_H + dH * h
        #         change_H -= (change_H > 180) * 180
        #
        #         for s in range(4):
        #             change_S = image_S * (1 - s * dS)
        #             for v in range(3):
        #                 change_V = image_V * (1 - v * dV)
        #                 img_change = cv2.merge([change_H, change_S, change_V])
        #                 img_change = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        #                 # change_path = save_path + '/' + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)
        #                 # if not os.path.exists(change_path):
        #                 #     os.makedirs(change_path)
        #                 img_name = os.path.splitext(os.path.basename(image))[0]
        #
        #                 # print(save_path)
        #                 # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
        #                 cv2.imwrite(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)+'.jpg', img_change)

    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths=[]
    aug_labels=[]

    classlist=os.listdir(aug_dir)
    for klass in classlist:
        classpath=os.path.join(aug_dir, klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries=pd.Series(aug_fpaths, name='filepaths')
    Lseries=pd.Series(aug_labels, name='labels')
    aug_df=pd.concat([Fseries, Lseries], axis=1)
    df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df
def initial_balance_train(df, n, working_dir, img_size):
    df = df.copy()
    # df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'balanced_train3')  # directory to store augmented images
    # if os.path.isdir(aug_dir):# start with an empty directory
    #     shutil.rmtree(aug_dir)
    # os.mkdir(aug_dir)
    # for label in df['labels'].unique():
    #     dir_path=os.path.join(aug_dir,label)
    #     os.mkdir(dir_path) # make class directories within aug directory
    # # create and store the augmented images
    # total = 0
    # gen = ImageDataGenerator(vertical_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2],width_shift_range=.2,
    #                          height_shift_range=.2, fill_mode = 'constant')
    # groups = df.groupby('labels')  # group by class
    # group1 = groups.get_group(df['labels'].unique()[0])
    # group2 = groups.get_group(df['labels'].unique()[1])
    # groups_len = [len(group1), len(group2)]
    # groups_diff = abs(len(group2) - len(group1))
    # if groups_diff != 0:
    #
    #     balance_len = min(groups_len)
    #     balance_len_index = groups_len.index(balance_len)
    #     print(len(group1), len(group2))
    #     print(balance_len, balance_len_index, groups_diff, df['labels'].unique()[balance_len_index])
    #
    #     group = groups.get_group(
    #         df['labels'].unique()[balance_len_index])  # a dataframe holding only rows with the specified label
    #     # determine how many samples there are in this class
    #     # print(sample_count)
    #     # exit()
    #     # if the class has less than target number of images
    #     aug_img_count = 0
    #     delta = groups_diff  # number of augmented images to create
    #
    #     target_dir = os.path.join(aug_dir, df['labels'].unique()[balance_len_index])  # define where to write the images
    #     print(target_dir)
    #     msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', df['labels'].unique()[
    #         balance_len_index], str(delta))
    #
    #     print(msg, '\r', end='')  # prints over on the same line
    #
    #     aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
    #                                       class_mode=None, batch_size=1, shuffle=False,
    #                                       save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
    #                                       save_format='jpg')
    #     while aug_img_count < delta:
    #
    #         images = next(aug_gen)
    #         aug_img_count += len(images)


    aug_fpaths = []
    aug_labels = []

    classlist = os.listdir(aug_dir)
    for klass in classlist:
        classpath = os.path.join(aug_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))

    return df
def initial_balance_val(df, n, working_dir, img_size):
    df = df.copy()
    # df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'balanced_val3')  # directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total = 0
    gen = ImageDataGenerator(horizontal_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups = df.groupby('labels')  # group by class
    # print(df['labels'].unique())
    group1 = groups.get_group(df['labels'].unique()[0])
    group2 = groups.get_group(df['labels'].unique()[1])
    groups_len = [len(group1), len(group2)]
    groups_diff = abs(len(group2) - len(group1))
    # print(df['labels'].unique()[balance_len_index])
    # exit()
    if groups_diff != 0:

        balance_len = min(groups_len)
        balance_len_index = groups_len.index(balance_len)
        # print(len(group1), len(group2))
        print(balance_len, balance_len_index, groups_diff, df['labels'].unique()[balance_len_index])

        group = groups.get_group(
            df['labels'].unique()[balance_len_index])  # a dataframe holding only rows with the specified label
        # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        # if the class has less than target number of images
        aug_img_count = 0
        delta = groups_diff  # number of augmented images to create

        target_dir = os.path.join(aug_dir, df['labels'].unique()[balance_len_index])  # define where to write the images
        print(target_dir)
        msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', df['labels'].unique()[
            balance_len_index], str(delta))

        print(msg, '\r', end='')  # prints over on the same line

        aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
                                          class_mode=None, batch_size=1, shuffle=False,
                                          save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                          save_format='jpg')
        while aug_img_count < delta:

            images = next(aug_gen)
            aug_img_count += len(images)


        aug_fpaths = []
        aug_labels = []

        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
        print('Length of augmented dataframe is now ', len(df))
    return df

def initial_balance_test(df, n, working_dir, img_size):
    df = df.copy()
    # df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'balanced_test3')  # directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total = 0
    gen = ImageDataGenerator(horizontal_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups = df.groupby('labels')  # group by class
    group1 = groups.get_group(df['labels'].unique()[0])
    group2 = groups.get_group(df['labels'].unique()[1])
    groups_len = [len(group1), len(group2)]
    groups_diff = len(group2) - len(group1)
    if groups_diff != 0:

        balance_len = min(groups_len)
        balance_len_index = groups_len.index(balance_len)
        print(len(group1), len(group2))
        print(balance_len, balance_len_index, groups_diff, df['labels'].unique()[balance_len_index])

        group = groups.get_group(
            df['labels'].unique()[balance_len_index])  # a dataframe holding only rows with the specified label
        # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        # if the class has less than target number of images
        aug_img_count = 0
        delta = groups_diff  # number of augmented images to create

        target_dir = os.path.join(aug_dir, df['labels'].unique()[balance_len_index])  # define where to write the images
        print(target_dir)
        msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', df['labels'].unique()[
            balance_len_index], str(delta))

        print(msg, '\r', end='')  # prints over on the same line

        aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
                                          class_mode=None, batch_size=1, shuffle=False,
                                          save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                          save_format='jpg')
        while aug_img_count < delta:

            images = next(aug_gen)
            aug_img_count += len(images)


        aug_fpaths = []
        aug_labels = []

        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
        print('Length of augmented dataframe is now ', len(df))

    return df


def add_s_p_noise(img):
    # Getting the dimensions of the image
    row, col, _ = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img

def add_s_p_noise_testing(img):
    # Getting the dimensions of the image
    row, col, _ = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(1000, 5000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(1000, 5000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img
def rotate_image(image):
    return np.rot90(image, np.random.choice([-2, -1, 0, 1, 2]))

from torchvision.utils import save_image
def save_noisy_image(img, name):
    if img.size(1) == 3:
        img = img.view(img.size(0), -1)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 1, 28, 28)
        save_image(img, name)
def create_testing_dataset(classpath,working_dir='./'):
    flist = [f for f in os.listdir(classpath) if not f.startswith('.')]
    # print(flist)
    # exit()
    print(f'original length: {len(flist)}')
    aug_dir=os.path.join(working_dir, 'aug_testing_dataset_no_stretches/Others/')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)

    # create and store the augmented images

    '(1) Random rotation by a multiple of 90 degrees, ' \
    '(2) Random rotation in the range of-45 to 45 degrees,' \
    ' (3) Translation, (4) Reflection, (5) Shear, (6) Hue jitter, (7) Saturation jitter, (8) Brightness jitter, ' \
    '(9) Contrast jitter, (10) Salt and Pepper Noise, (11) Gaussian Noise, (12) Synthetic Blur and (13) Scaling.'


     # group by class



    for image in flist:
        print(f'########################Augmenting {image}##########################')

        image = os.path.join(classpath, image)

        img = Image.open(image).convert("RGB")
        img_name = os.path.splitext(os.path.basename(image))[0]
        gen = ImageDataGenerator(vertical_flip=True, rotation_range=25, zoom_range=[1.3, 1.7],
                                 brightness_range=[0.3, 1.7], channel_shift_range=10)

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
        angle = [70,140,210,280,350]
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

        # print(TF.to_tensor(img).unsqueeze_(0))
        # exit()
        # print(cv2.imread(image).shape)
        cv2.imwrite(aug_dir+'/' + img_name + '-testing-10.jpg',
                    add_s_p_noise_testing(cv2.imread(image)))


        orig_img = Image.open(image)
        blurred_img = T.GaussianBlur(kernel_size=(51, 91), sigma=4)(orig_img).convert('RGB')

        noise_img = add_noise(T.ToTensor()(orig_img), 0.2)
        noise_img = T.ToPILImage()(noise_img).convert('RGB')

        img_name = os.path.splitext(os.path.basename(image))[0]

        # print(save_path)
        # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
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
def new_balance(classpath,working_dir='./'):
    flist = [f for f in os.listdir(classpath) if not f.startswith('.')]
    # print(flist)
    # exit()
    print(f'original length: {len(flist)}')
    aug_dir=os.path.join(working_dir, 'aug_new_normal_2')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)

    # create and store the augmented images

    '(1) Random rotation by a multiple of 90 degrees, ' \
    '(2) Random rotation in the range of-45 to 45 degrees,' \
    ' (3) Translation, (4) Reflection, (5) Shear, (6) Hue jitter, (7) Saturation jitter, (8) Brightness jitter, ' \
    '(9) Contrast jitter, (10) Salt and Pepper Noise, (11) Gaussian Noise, (12) Synthetic Blur and (13) Scaling.'


     # group by class


    for image in flist:
        print(f'########################Augmenting {image}##########################')

        image = os.path.join(classpath, image)

        img = Image.open(image)
        img_name = os.path.splitext(os.path.basename(image))[0]

        ################################################################################################################################################
        angle = [90,180,270]
        rotated_img = img.rotate(random.choice(angle), expand=True)
        rotated_img=rotated_img.convert('RGB')
        rotated_img.save(
            aug_dir+'/' + img_name + '-01.jpg')
        ################################################################################################################################################
        transform = T.RandomRotation((-45,45))
        rotated_img = transform(img)
        rotated_img=rotated_img.convert('RGB')
        rotated_img.save(
            aug_dir+'/' + img_name + '-02.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=0,translate=(0.3,0))
        translated_img = transform(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-03.jpg')
        ################################################################################################################################################
        im_mirror = ImageOps.mirror(img)
        im_mirror=im_mirror.convert('RGB')
        im_mirror.save(aug_dir+'/' + img_name + '-04.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=20,shear=0.2)
        translated_img = transform(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-05.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(hue=.3)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-06.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(saturation=.3)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-07.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(brightness=.5)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-08.jpg')
        ################################################################################################################################################
        jitter = T.ColorJitter(contrast=.5)
        translated_img = jitter(img)
        translated_img=translated_img.convert('RGB')
        translated_img.save(
            aug_dir+'/' + img_name + '-09.jpg')
        ################################################################################################################################################

        # print(TF.to_tensor(img).unsqueeze_(0))
        # exit()
        # print(cv2.imread(image).shape)
        cv2.imwrite(aug_dir+'/' + img_name + '-10.jpg',
                    add_s_p_noise(cv2.imread(image)))


        orig_img = Image.open(image)
        blurred_img = T.GaussianBlur(kernel_size=(51, 91), sigma=3)(orig_img).convert('RGB')

        noise_img = add_noise(T.ToTensor()(orig_img), 0.1)
        noise_img = T.ToPILImage()(noise_img).convert('RGB')

        img_name = os.path.splitext(os.path.basename(image))[0]

        # print(save_path)
        # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
        blurred_img.save(
            aug_dir+'/' + img_name + '-12.jpg')
        noise_img.save(
            aug_dir+'/' + img_name + '-11.jpg')
        ################################################################################################################################################
        transform = T.RandomAffine(degrees=0,scale=(1.2,1.5))
        scaled_img = transform(img)
        scaled_img=scaled_img.convert('RGB')
        scaled_img.save(
            aug_dir+'/' + img_name + '-13.jpg')


def balance_train(df, n, working_dir, img_size):
    df=df.copy()
    df = add_blur_noise_train(df)
    print('Initial length of dataframe is ', len(df))
    aug_dir=os.path.join(working_dir, 'aug_train3')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total=0
    gen=ImageDataGenerator(rescale = 1./255, horizontal_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2])
    groups=df.groupby('labels') # group by class

    for label in df['labels'].unique():  # for every class
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        # print(sample_count)
        # exit()
        if sample_count< 10*sample_count: # if the class has less than target number of images
            aug_img_count=0
            delta=10*sample_count - sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe(group,  x_col='filepaths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
        ############################################################################

        save_path = target_dir
        # value_path = 'G:/UNet2/data/ECU/test/HSV/Value/change'
        # sat_path = 'G:/UNet2/data/ECU/test/HSV/Saturation/change'
        # image_names = group['filepaths']
        # dH = 30
        # dS = 0.2
        # dV = 0.2
        # # print(image_names)
        # for image in image_names:
        #     # Change_Value(image, image_path, value_path)
        #     # Change_Saturation(image, image_path, sat_path)
        #     # break
        #     # print()
        #     img = cv2.imread(image)
        #
        #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
        #     image_H, image_S, image_V = cv2.split(img_hsv)
        #     for h in range(3):
        #         change_H = image_H + dH * h
        #         change_H -= (change_H > 180) * 180
        #
        #         for s in range(4):
        #             change_S = image_S * (1 - s * dS)
        #             for v in range(3):
        #                 change_V = image_V * (1 - v * dV)
        #                 img_change = cv2.merge([change_H, change_S, change_V])
        #                 img_change = cv2.cvtColor(img_change.astype('uint8'), cv2.COLOR_HSV2BGR)
        #                 # change_path = save_path + '/' + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)
        #                 # if not os.path.exists(change_path):
        #                 #     os.makedirs(change_path)
        #                 img_name = os.path.splitext(os.path.basename(image))[0]
        #
        #                 # print(save_path)
        #                 # print(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v))
        #                 cv2.imwrite(save_path+'/' + img_name + 'H%dS%.1fV%.1f' % (dH * h, 1 - dS * s, 1 - dV * v)+'.jpg', img_change)

    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths=[]
    aug_labels=[]

    classlist=os.listdir(aug_dir)
    for klass in classlist:
        classpath=os.path.join(aug_dir, klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries=pd.Series(aug_fpaths, name='filepaths')
    Lseries=pd.Series(aug_labels, name='labels')
    aug_df=pd.concat([Fseries, Lseries], axis=1)
    df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df

def show_image_samples(gen):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())
    images,labels=next(gen) # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i] /255
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=14)
        plt.axis('off')
    plt.show()


def predictor(model, test_gen, test_steps):
    y_pred = []
    y_true = test_gen.labels
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds = model.predict(test_gen, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]  # labels are integer values
        if pred_index != true_index:  # a misclassification has occurred
            errors = errors + 1
        y_pred.append(pred_index)

    acc = (1 - errors / tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    if class_count <= 30:
        cm = confusion_matrix(ytrue, ypred)
        # plot the confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count) + .5, classes, rotation=90)
        plt.yticks(np.arange(class_count) + .5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)  # create classification report
    print("Classification Report:\n----------------------\n", clr)
    return errors, tests

def main():
    new_balance('new2_normal')
    exit()
    # sdir = r'Original_Images/Original_Images/'
    sdir = 'Original_Images/Original_Images/'
    # sdir = '/Users/timothy/Desktop/Timothy-2021-2022/health_ai/SemanticSegmentation/aug_testing_dataset_no_stretches_restored'

    sdir_train = r'Augmented Images/Augmented Images/'

    # sdir_train = 'aug_testing_dataset'
    filepaths_train = []
    labels_train = []
    classlist_train = os.listdir(sdir_train)
    print(classlist_train)
    classlist_train = [classlist_train[0],classlist_train[2]]
    print(classlist_train)
    for klass in classlist_train:
        classpath = os.path.join(sdir_train, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths_train.append(fpath)
            labels_train.append(klass)
    Fseries_train = pd.Series(filepaths_train, name='filepaths')
    Lseries_train = pd.Series(labels_train, name='labels')
    train_df = pd.concat([Fseries_train, Lseries_train], axis=1)



    filepaths = []
    labels = []
    classlist = os.listdir(sdir)
    print(classlist)
    classlist = classlist[1:]
    print(classlist)
    for klass in classlist:
        classpath = os.path.join(sdir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    print(train_df)
    print(df)

    # exit()
    train_df = initial_balance_train(train_df, 200, r'./', (224, 224))
    df = initial_balance_df(df, 200, r'./', (224, 224))
    # df = balance_df(df, 200, r'./', (224, 224))
    # train_df, dummy_df = train_test_split(train_df, train_size=1, shuffle=True, random_state=13, stratify=df['labels'])
    valid_df, test_df = train_test_split(df, train_size=.65, shuffle=True, random_state=13,
                                         stratify=df['labels'])
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))

    # get the number of classes and the images count for each class in train_df
    classes = sorted(list(train_df['labels'].unique()))
    class_count = len(classes)
    print('The number of classes in the dataset is: ', class_count)
    groups = train_df.groupby('labels')
    print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
    countlist = []
    classlist = []
    for label in sorted(list(train_df['labels'].unique())):
        group = groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)
        print('{0:^30s} {1:^13s}'.format(label, str(len(group))))
    #
    # # get the classes with the minimum and maximum number of train images
    # max_value = np.max(countlist)
    # max_index = countlist.index(max_value)
    # max_class = classlist[max_index]
    # min_value = np.min(countlist)
    # min_index = countlist.index(min_value)
    # min_class = classlist[min_index]
    # print(max_class, ' has the most images= ', max_value, ' ', min_class, ' has the least images= ', min_value)
    # lets get the average height and width of a sample of the train images
    # ht = 0
    # wt = 0
    # select 100 random samples of train_df
    # train_df_sample = train_df.sample(n=100, random_state=123, axis=0)
    # for i in range(len(train_df_sample)):
    #     fpath = train_df_sample['filepaths'].iloc[i]
    #     img = plt.imread(fpath)
    #     shape = img.shape
    #     ht += shape[0]
    #     wt += shape[1]
    # print('average height= ', ht // 100, ' average width= ', wt // 100, 'aspect ratio= ', ht / wt)

    ##########################################################################################################################
    n = 200  # number of samples in each class
    working_dir = r'./'  # directory to store augmented images
    img_size = (224, 224)  # size of augmented images
    # train_df = initial_balance_train(train_df, n, working_dir, img_size)
    # test_df = initial_balance_test(test_df, n, working_dir, img_size)
    # valid_df = initial_balance_val(valid_df, n, working_dir, img_size)
    # # exit()
    #
    # train_df = balance_train(train_df, n, working_dir, img_size)
    # print(f'Training df',train_df,sep='\n')
    # test_df = balance_test(test_df, n, working_dir, img_size)
    # print(f'Test df', test_df, sep='\n')
    # valid_df = balance_val(valid_df, n, working_dir, img_size)
    # print(f'Valid df', valid_df, sep='\n')
    # print(train_df)
    # exit()
    ##########################################################################################################################
    batch_size = 64  # We will use and EfficientetB3 model, with image size of (200, 250) this size should not cause resource error
    trgen = ImageDataGenerator()
    t_and_v_gen = ImageDataGenerator()
    # msg = '{0:70s} for train generator'.format(' ')
    # print(msg, '\r', end='')  # prints over on the same line
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical', color_mode='rgb', shuffle=True,
                                          batch_size=batch_size)
    # print(train_gen)
    # # exit()
    # msg = '{0:70s} for valid generator'.format(' ')
    # print(msg, '\r', end='')  # prints over on the same line
    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                                class_mode='categorical', color_mode='rgb', shuffle=False,
                                                batch_size=batch_size)
    # # # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
    # # # this insures that we go through all the sample in the test set exactly once.
    length = len(df)
    test_batch_size = \
    sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
    test_steps = int(length / test_batch_size)
    # msg = '{0:70s} for test generator'.format(' ')
    # print(msg, '\r', end='')  # prints over on the same line
    # for image in test_df:

    test_gen = t_and_v_gen.flow_from_dataframe(df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical', color_mode='rgb', shuffle=False,
                                               batch_size=test_batch_size)
    # # from the generator we can get information we will need later
    # classes = list(train_gen.class_indices.keys())
    # class_indices = list(train_gen.class_indices.values())
    # class_count = len(classes)
    # labels = test_gen.labels
    # print('test batch size: ', test_batch_size, '  test steps: ', test_steps, ' number of classes : ', class_count)
    #
    # ##########################################################################################################################
    # # show_image_samples(train_gen)
    # ##########################################################################################################################
    img_shape = (img_size[0], img_size[1], 3)
    # model_name = 'EfficientNetB3'
    #

    base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet',
                                                                   input_shape=img_shape, pooling='max')
    # # loaded_model = tf.keras.load_model('monkey pox_100.0.h5')
    # # # Note you are always told NOT to make the base model trainable initially- that is WRONG you get better results leaving it trainable
    # # base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=img_shape,
    # #                                                   pooling='max')
    base_model.trainable = True
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=.4, seed=13)(x)
    output = Dense(class_count, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    lr = .001  # start with this learning rate
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    # # ##########################################################################################################################
    epochs = 40
    ask_epoch = 80
    ask = LR_ASK(model, epochs, ask_epoch)
    # # rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,verbose=1)
    # # callbacks=[rlronp, ask]
    callbacks = [ask]
    # ##########################################################################################################################
    history = model.fit(x=train_gen, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=valid_gen,
                        validation_steps=None, shuffle=False, initial_epoch=0)
    #########################################################################################################################
    model = tf.keras.models.load_model(
        'monkey pox_100.0.h5')   #my_new_balanced_4batch
    errors, tests = predictor(model, test_gen, test_steps)
    ##########################################################################################################################
    subject = 'monkey pox'
    acc = str((1 - errors / tests) * 100)
    index = acc.rfind('.')
    acc = acc[:index + 3]
    print(acc)
    save_id = subject + '_' + str(acc) + 'my_new_balanced_5.h5'
    model_save_loc = os.path.join(working_dir, save_id)
    model.save(model_save_loc)
    print('model was saved as ', model_save_loc)

if __name__ == "__main__":
    main()