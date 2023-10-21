# AICOM-MP(Quantized)
![](AICOM_APP_architecture%20(1).png)


# 1. Download EfficientNetLite Model
```
# execute on command line
pip3 install git+https://github.com/sebastian-sz/efficientnet-lite-keras@main
```
## Disclaimer from the owner of the package
* This is a package with EfficientNet-Lite model variants adapted to Keras.  
* The model's weights are converted from [original repository](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/).
* Check out their [repo](https://github.com/sebastian-sz/efficientnet-lite-keras) for more details.


# 2. EfficientNetLite Model Selection

### Input shapes
The following table shows input shapes for each model variant:

| Model variant | Input shape |
|:-------------:|:-----------:|
|       B0      | `224,224`   |
|       B1      | `240,240`   |
|       B2      | `260,260`   |
|       B3      | `280,280`   |
|       B4      | `300,300`   |

* Since we aim to deploy ML model on **resource-constrained mobile devices**, our model accpets input images of size **224 x 224** in this Monkeypox case study.
* We selected **EfficientNetLiteB0** as the base model.

# 3. Building AICOM-MP Model for Resource-constrained Mobile Devices
1. `AICOM_MP_Lite.py`
  * EfficientNetB0 training and fine-tuning
2. `AICOM_MP_QT.py`
  * Calibration through 200 representative dataset
    * `representative_dataset()`
    * `converter.representative_dataset = representative_dataset`
  * INT-8 Quantization
    * `converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
3. More information refers to [TensorflowLite Quantization Website](https://www.tensorflow.org/lite/performance/post_training_quantization)
  

# 4. Testing Model Performance

## FLOPS
* `FLOPs.py`
  * Calculating FLOPs for tflite model ([reference](https://github.com/lisosia/tflite-flops/tree/main))
    * `tflite_flops.calc_flops("/Monkeypox/AICOM_MP_INT8_200rep.tflite")` 
  * Calculating FLOPs for EfficientNet model ([reference](https://github.com/tokusumi/keras-flops/blob/master/notebooks/flops_calculation_tfkeras.ipynb))
    * `model = tf.keras.models.load_model(
        '../AICOM_MP_weight.h5')
    flops = get_flops(model)`

## Classification Performance on AICOM-MP dataset (312 images)
* `AICOM_MP_QT_Model_Testing.py`