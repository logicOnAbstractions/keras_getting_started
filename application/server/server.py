""" serving keras on flask webserver """

from keras.applications import imagenet_utils
from PIL import Image                       # consider replacing pillow by scikit
from flask import Flask, request, jsonify
import io
from main import MainProgram
from utils import *
from logger import get_root_logger

LOG = get_root_logger(BASE_LOGGER_NAME)


app             = Flask(__name__)
model           = None
mp              = None


############## api endpoints ##############
@app.route("/")
def home():
    LOG.info(f"/home ping... ")
    return "Welcome to the ML api (Keras)"

@app.route("/runmodel")
def run_model():
    LOG.info(f"")
    try:
        value = request.args.get("model")
    except Exception as ex:
        LOG.error(f"Failed to get model argument: {request.args} (expected model key)")

    if value == "MNIST":
        mp.execute_all()

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
            prediction = mp.predict(image)

        # return the data dictionary as a JSON response
        return jsonify(prediction)
    else:
        raise Exception(f" invalid prediction - wrong input data or error in model setup.")

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    mp = MainProgram()
    app.run()
