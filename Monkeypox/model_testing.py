import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def predictor(model, test_gen):
    y_pred = []
    y_true = test_gen.labels
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds = model.predict(test_gen, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]
        if pred_index != true_index:
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
    sdir = 'coco/'
    filepaths = []
    labels = []
    classlist = os.listdir(sdir)
    print(classlist)
    classlist = classlist[:2]
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
    print(f'Testing dataset: {len(df)} images')
    print(df)
    img_size = (224, 224)
    t_and_v_gen = ImageDataGenerator()
    length = len(df)
    test_batch_size = \
    sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
    test_gen = t_and_v_gen.flow_from_dataframe(df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical', color_mode='rgb', shuffle=False,
                                               batch_size=test_batch_size)
    model = tf.keras.models.load_model(
        'monkey pox_100.0my_new_balanced_3_batch_size_48_3:54.h5')
    errors, tests = predictor(model, test_gen)
    subject = 'monkey pox'
    acc = str((1 - errors / tests) * 100)
    index = acc.rfind('.')
    acc = acc[:index + 3]
    print(acc)

if __name__ == "__main__":
    main()
