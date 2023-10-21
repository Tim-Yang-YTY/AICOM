

from thop import profile, clever_format

import time
import tensorflow as tf
from keras_flops import get_flops

#python3 FLOPs.py --config-file
import tflite_flops

if __name__ == '__main__':
    tflite_flops.calc_flops("/Monkeypox/AICOM_MP_INT8_200rep.tflite")
    model = tf.keras.models.load_model(
        '../AICOM_MP_weight.h5')
    flops = get_flops(model)
    print(f"FLOPS: {flops / 10 ** 6:.6} M")