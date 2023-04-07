import numpy as np
import tensorflow as tf
import PIL.Image

def predict_image(model_dirpath, image_filepath):
    model = tf.saved_model.load(str(model_dirpath))
    serve = model.signatures['serving_default']
    input_shape = serve.inputs[0].shape[1:3]

    image = PIL.Image.open(image_filepath).resize(input_shape)
    input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
    input_array = input_array[:, :, :, (2, 1, 0)]  # => BGR
    input_tensor = tf.convert_to_tensor(input_array)

    output = serve(input_tensor)
    output = list(output.values())[0].numpy()

    return output

def preprocess_image(image_filepath, target_size):
    image = PIL.Image.open(image_filepath)

    # Resize the image
    image = image.resize(target_size)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the array to a TensorFlow tensor
    tensor = tf.convert_to_tensor(image_array)

    # Expand the dimensions of the tensor to add a batch dimension
    tensor = tf.expand_dims(tensor, axis=0)

    # Convert the data type of the tensor to float32
    tensor = tf.cast(tensor, tf.float32)

    # Reverse the order of the color channels from RGB to BGR
    tensor = tensor[:, :, :, ::-1]

    return tensor
