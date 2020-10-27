""" the main keras-class that manage the brunt of the work wrt to the model etc. """

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from kerastuner.tuners import RandomSearch
import keras
import kerastuner.engine.hyperparameters as hp
from application.engine.layers_descriptions import DefaultArch
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
        LOG.info(f"Instantiating {self.__class__.__name__}")
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
        """ uses content of self.configs to instantiate all of our model. this uses the functional notation, which is fine
         for non-tuners model, e.g. the default (see Tuner.instantiate_layers() override for details). """
        self.model = DefaultArch(self.configs)()
        self.model.summary()


class DigitsMNIST(Model):
    """ sample model that trains to recognize the MNIST digits classic example """

    def __init__(self, configs):
        """ """
        super().__init__(configs)
        LOG.info(f"Instantiating {self.__class__.__name__}")
        self.dao            = DiskDao()
        self.layers         = DefaultArch(configs)
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


class Tuner(Model):
    """ a thing to fool around & test syntaxes etc.

        currently we test for the keras tuner thingy, so this models makes default choiecs
        to make that happen in the layer choices etc.
     """

    def __init__(self, configs):
        """ """
        super().__init__(configs)
        LOG.info(f"Instantiating {self.__class__.__name__}")

        self.dao            = DiskDao()
        self.tuner_model    = DefaultArch(configs)
        self.configs        = configs
        self.mode           = "default"         # TODO: for now
        # self.model          = None
        # self.instantiante_layers()

    def train(self):
        """ TODO: actually this should be def tune(..) - will have to reoncile/fix the nomentalures at some point """
        tuner = RandomSearch(self.tuner_model, objective='val_accuracy', max_trials=3, executions_per_trial=2,
                             directory='my_dir', project_name='helloworld')

        # proprocess the data
        data = self.dao.get_mnist_dataset()         # TODO: currently returns none
        LOG.info(f"BUILD: data from keras: {data.keys()}")
        # pass it to our model - the model also takes care of preprocessing so we just pass it the raw data we loaded
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        x = data["x_train"]
        y = data["y_train"]
        x_val = data["x_test"]
        y_val = data["y_test"]

        print(f"label shapes: {y.shape} {y_val.shape}")
        # call the tuner on that

        try:
            tuner.search(x, y, epochs=2, validation_data=(x_val, y_val))
        except Exception as ex:
            LOG.error(f"Failed to tuner.search: {ex}")
        models = tuner.get_best_models(num_models=2)

        # self.tuner_model.fit(data["x_train"],data["y_train"])
        LOG.info(f"Finished tuning model. Summary:")
        tuner.results_summary()

    def predict_single(self, data):
        """ makes a prediction, assuming a trained model. if not will return random answer """

    def instantiante_layers(self):
        """ known issue: cannot build (e.g. call the functional keras.Layer()() form ) right away.
            the kerastuner expects the model to build only at runtime, after we pass it kerastuner.engines.hyperparameters (hp))
        """
        self.tuner_model = DefaultArch(self.configs)        # NOTE: we DONT use the functional notation here, because the actual model needs to be built at runtime when self.tuner.build(hp) is called by the kerastuner
        self.tuner_model.summary()


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