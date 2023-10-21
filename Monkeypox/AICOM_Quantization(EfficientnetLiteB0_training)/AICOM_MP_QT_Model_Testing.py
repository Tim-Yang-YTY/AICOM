import os
import time

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def run_tflite_model(tflite_file, test_image):

    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    predictions = predictions.argmax()
    return predictions


def main():
    converted_model = "AICOM_MP_INT8_200rep.tflite"
    filepaths = []
    labels = []
    testing_dir = "../Original_Images/Original_Images"
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
    predictions = []
    inference_times = []
    for i in range(len(filepaths)):
        img = cv2.imread(filepaths[i])
        # img = bytearray(img)
        resized = cv2.resize(img, (224, 224))
        resized = resized.astype(np.float32)
        # resized = resized / 255.
        test_image = np.expand_dims(resized, axis=0)
        # print(test_image.shape)
        prediction = run_tflite_model(converted_model, test_image)
        predictions.append(prediction)

        print(i)
        if prediction == labels[i]:
            score += 1
    print(score, score / len(filepaths))
    cm = confusion_matrix(labels, predictions)
    clr = classification_report(labels, predictions, target_names=classlist, digits=4)  # create classification report
    print("Confusion Matrix:\n----------------------\n", cm)
    print("Classification Report:\n----------------------\n", clr)

if __name__ == "__main__":
    main()