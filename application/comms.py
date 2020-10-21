""" api rsts & such - communications """
# import the necessary packages
import requests
import json

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
# r = requests.post(KERAS_REST_API_URL, files=payload).json()
r = requests.post(KERAS_REST_API_URL, files=payload)

# ensure the request was successful
if r.status_code==200:
    # loop over the predictions and display them
    payload = json.loads(r.text)
    for (i, result) in enumerate(payload["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print(f"Request failed {r.content}")
