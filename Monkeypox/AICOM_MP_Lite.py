import os

import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
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
from torchvision.utils import save_image
import tensorflow as tf
from skimage import io
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, classification_report


def initial_balance_train(df, n, working_dir, img_size):
    df = df.copy()
    print('Initial length of dataframe is ', len(df))
    aug_dir = os.path.join(working_dir, 'balanced_train3')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path)
    gen = ImageDataGenerator(vertical_flip=True,  rotation_range=20, zoom_range=[.99, 1.01],brightness_range=[0.8, 1.2],width_shift_range=.2,
                             height_shift_range=.2, fill_mode = 'constant')
    groups = df.groupby('labels')
    group1 = groups.get_group(df['labels'].unique()[0])
    group2 = groups.get_group(df['labels'].unique()[1])
    groups_len = [len(group1), len(group2)]
    groups_diff = abs(len(group2) - len(group1))
    if groups_diff != 0:

        balance_len = min(groups_len)
        balance_len_index = groups_len.index(balance_len)
        print(len(group1), len(group2))
        print(balance_len, balance_len_index, groups_diff, df['labels'].unique()[balance_len_index])
        group = groups.get_group(
            df['labels'].unique()[balance_len_index])
        aug_img_count = 0
        delta = groups_diff

        target_dir = os.path.join(aug_dir, df['labels'].unique()[balance_len_index])
        print(target_dir)
        msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', df['labels'].unique()[
            balance_len_index], str(delta))

        print(msg, '\r', end='')
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

def representative_dataset():
    sdir_train = r'Augmented Images/Augmented Images/'
    filepaths_train = []
    labels_train = []
    classlist_train = [f for f in os.listdir(sdir_train) if not f.startswith('.')]
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


    img_size = (224, 224)
    batch_size = 200
    trgen = ImageDataGenerator()
    train_gen = trgen.flow_from_dataframe(train_df[:200], x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical', color_mode='rgb', shuffle=True,
                                          batch_size=batch_size)
    print("here")
    for i in range(200):
        print(i)
        yield[np.array(train_gen.next()[0], dtype=np.float32)]


def run_tflite_model(tflite_file, test_image):

    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    start_time = time.time()
    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    start_time = time.time()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    stop_time = time.time()
    predictions = predictions.argmax()
    inference_time = stop_time - start_time
    return predictions, inference_time*1000

def main():
    # base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights='imagenet',
    #                                                                input_shape=(224, 224, 3), pooling='max')
    # base_model.trainable = True
    # x = base_model.output
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    # x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
    #           bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    # x = Dropout(rate=.4, seed=13)(x)
    # output = Dense(2, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=output)
    # model.load_weights("/Users/timothy/Desktop/Timothy-2021-2022/health_ai/AICOM/Monkeypox/AICOM_MP_weight.h5")
    # model.summary()
    # exit()
    model = tf.keras.models.load_model('/Users/timothy/Desktop/Timothy-2022-2023/AICOM_qt')
    # print(loaded_model.summary())
    # exit()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    #
    # # Save
    with open("AICOM_MP_INT8_200rep.tflite", "wb") as file:
        file.write(tflite_model)

    exit()
    converted_model = "/Users/timothy/Desktop/Timothy-2021-2022/health_ai/AICOM/Monkeypox/AICOM_MP_INT_8_2.tflite"
    filepaths = []
    labels = []
    testing_dir = "/Users/timothy/Desktop/Timothy-2021-2022/health_ai/AICOM/Monkeypox/Original_Images/Original_Images"
    classlist = [f for f in os.listdir(testing_dir) if not f.startswith('.')]
    print(classlist)
    for klass in classlist:
        classpath = os.path.join(testing_dir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(classlist.index(klass))
    score = 0
    predictions=[]
    inference_times = []
    for i in range(len(filepaths)):
        img = cv2.imread(filepaths[i])
        # img = bytearray(img)
        resized = cv2.resize(img, (224, 224))
        resized = resized.astype(np.float32)
        # resized = resized / 255.
        test_image = np.expand_dims(resized, axis=0)
        # print(test_image.shape)
        prediction, inference_time = run_tflite_model(converted_model, test_image)
        predictions.append(prediction)
        inference_times.append(inference_time)
        print(i)
        if prediction == labels[i]:
            score += 1
    print(score,score/len(filepaths),sum(inference_times)/len(filepaths))

    class_count = len(classlist)
    cm = confusion_matrix(labels, predictions)
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    # plt.xticks(np.arange(class_count) + .5, classlist, rotation=90)
    # plt.yticks(np.arange(class_count) + .5, classlist, rotation=0)
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()
    clr = classification_report(labels, predictions, target_names=classlist, digits=4)  # create classification report
    print("Confusion Matrix:\n----------------------\n",cm)
    print("Classification Report:\n----------------------\n", clr)

    exit()





    # loaded_model_from_h5 = tf.keras.models.load_model('AICOM_MP_qt.tflite')  # Loading the H5 Saved Model
    # print(loaded_model_from_h5.summary())

    # model = tf.keras.models.load_model('AICOM_MP_weight.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # # tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.allow_custom_ops = True
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # # , tf.lite.OpsSet.SELECT_TF_OPS
    # tflite_model_quant = converter.convert()
    #
    # # convert
    # # tflite_model = tflite_converter.convert()
    # open("AICOM_MP_lite_model_quant.tflite", "wb").write(tflite_model_quant)


if __name__ == "__main__":
    main()