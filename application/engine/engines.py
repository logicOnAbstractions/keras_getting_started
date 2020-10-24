""" the main keras-class that manage the brunt of the work wrt to the model etc. """

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from application.engine.preprocessing import ImagePreprocessor
from application.engine.layers_descriptions import Default
from dao import DiskDao
import tensorflow as tf
import numpy as np
from utils import *
from logger import get_root_logger
import yaml

LOG = get_root_logger(BASE_LOGGER_NAME)

class Model:
    """ parent class, meant to be subclasses by specific implt. models
        will allow flexibility in using various models to make predictions
    """
    def __init__(self, configs):
        """ """
        self.configs        = configs
        self.preprocessor   = None
        self.model          = None

    def predict_single(self, data):
        """ makes a prediction. subclasses define what is predicted """
        raise NotImplemented

    def load_model(self):
        """ Loads the appropriate model from keras """
        raise NotImplemented
    def instantiante_layers(self):
        """ uses content of self.configs to instantiate all of our model """
        self.model = Default(self.configs)()
        self.model.summary()

class DigitsMNIST(Model):
    """ sample model that trains to recognize the MNIST digits classic example """

    def __init__(self, configs):
        """ """
        super().__init__(configs)
        self.dao            = DiskDao()
        self.layers         = Default(configs)
        self.configs        = configs
        self.mode           = "default"         # TODO: for now
        self.instantiante_layers()

    def train(self):
        """ launches all the steps necessary to preprocess data, make predictions, etc. """

        # proprocess the data
        data = self.dao.get_mnist_dataset()         # TODO: currently returns none
        LOG.info(f"got data from keras: {data.keys()}")
        # pass it to our model - the model also takes care of preprocessing so we just pass it the raw data we loaded
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        self.model.fit(data["x_train"],data["y_train"])
        LOG.info(f"Finished training model. {self.model.history}")

    def predict_single(self, data):
        """ makes a prediction, assuming a trained model. if not will return random answer """

class DogBreedModel(Model):
    """ takes in a dog image, & predicts what breed this is """
    def __init__(self):
        """ """
        super().__init__()
        self.target_size    = (224, 224)
        self.model          = self.load_model()

    def predict(self, image):
        """ takes in a image & predict the dog bread. expects a PIL image"""

        # preprocess the image and prepare it for classification
        answer = {"success":False}
        image = self.prepare_image(image)

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = self.model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        answer["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            answer["predictions"].append(r)

        # indicate that the request was a success
        answer["success"] = True
        return answer

    def prepare_image(self, image):
        """
        :param image: obj
        :param target_size: the dims of the image we use
        :return: a resized image in rgb
        """

        # if the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(self.target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # return the processed image
        return image

    def load_model(self ):
        """ ResNet50 is our model in this case """
        model = ResNet50(weights="imagenet")
        return model