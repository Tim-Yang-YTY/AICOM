import tensorflow as tf
import time
from PIL import Image

def prediction(modelPath, imagePath):
	# Load the TensorFlow SavedModel
	model = tf.saved_model.load(modelPath)

	# Load the class labels from a text file
	with open(f"{modelPath}/labels.txt", 'r') as f:
		class_labels = f.read().splitlines()

	# Get the prediction function from the model
	predict_fn = model.signatures["serving_default"]

	# Load an image file
	image = Image.open(imagePath)
	image = image.resize((224, 224))

	# Preprocess the image
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = tf.expand_dims(image, axis=0)
	image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

	# Make a prediction
	start_time = time.time()
	prediction = predict_fn(image)
	end_time = time.time()

	time_taken = round((end_time - start_time) * 1000, 2)
	
	# Get the predicted class label
	class_index = tf.argmax(prediction['outputs'], axis=-1).numpy()[0]
	class_label = class_labels[class_index]

	class_probs = {}

	for i in range(len(class_labels)):
		class_label = class_labels[i]
		score = prediction['outputs'][0][i]
		percentage = score * 100
		print(f"{class_label}: {percentage:.2f}%")
		class_probs[class_label] = f"{percentage:.2f}%"

	print(class_probs['Others'])
	return class_probs['Monkeypox'], class_probs['Others'], time_taken
