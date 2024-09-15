import requests
import time
import json

def predict_image(image_path, resources, projectId, iteration, predictionKey):
    with open(image_path, 'rb') as f:
        image_data = f.read()

    url = f"https://{resources}/customvision/v3.0/Prediction/{projectId}/classify/iterations/{iteration}/image"
    headers= {'Content-Type': 'application/octet-stream','Prediction-Key': predictionKey}

    # Sending post request
    start_time = time.time()
    response = requests.post(url, headers=headers, data=image_data)
    end_time = time.time()

    time_taken = round((end_time - start_time) * 1000, 2)

    if response.status_code == 200:
        data = response.json()
        class_probabilities = {}

        for prediction in data["predictions"]:
            class_name = prediction["tagName"]
            class_probability = prediction["probability"]
            class_probabilities[class_name] = "{:.2f}".format(round(class_probability * 100, 2))

        # Extract the Monkeypox probability and remove it from the dictionary
        monkeypox_probability = class_probabilities.pop("Monkeypox", None)
        if monkeypox_probability is not None:
            print(f"Monkeypox probability: {monkeypox_probability}%")

        # Sort the remaining classes by their probabilities
        sorted_classes = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Print the probabilities for the remaining classes
        for class_name, probability in sorted_classes:
            print(f"{class_name} probability: {probability}%")

        # Print the time taken
        print(f"Time taken: {time_taken} milliseconds")
        
        # Return the Monkeypox and Others probabilities and the time taken
        return f"{monkeypox_probability}%", f"{class_probabilities.get('Others')}%",time_taken

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, None, time_taken
