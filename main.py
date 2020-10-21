""" contains the base functions

TODO: I should do a generic "MainProgram" boilertplate. e.g. a class structure as for RT, Docker stuff etc.

actually, that MP should also include basic stuff like loading json/yaml configs, storing those somewhere, etc.

"""

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image                       # consider replacing pillow by scikit
import numpy as np
from flask import Flask, request, jsonify
from utils import load_default_model, prepare_image
import io


app = Flask(__name__)
model = None
# model = load_default_model()

############## api endpoints ##############
@app.route("/")
def home():
    return "Welcome to the ML api (Keras)"


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target_size=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model = load_default_model()
    app.run()
