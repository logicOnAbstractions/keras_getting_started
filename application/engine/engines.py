""" the main keras-class that manage the brunt of the work wrt to the model etc. """

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np

class Model:
    """ parent class, meant to be subclasses by specific implt. models
        will allow flexibility in using various models to make predictions
    """
    def __init__(self):
        """ """

    def predict_single(self, data):
        """ makes a prediction. subclasses define what is predicted """
        raise NotImplemented

    def load_model(self):
        """ Loads the appropriate model from keras """
        raise NotImplemented

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