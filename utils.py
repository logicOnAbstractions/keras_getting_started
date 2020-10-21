""" basic methods, csts, etc. """

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
###### constants

###### methods

def load_default_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    model = ResNet50(weights="imagenet")
    return model

def load_custom_model():
    """ same as above but for our own learning model """

def prepare_image(image, target_size):
    """
    :param image: obj 
    :param target_size: the dims of the image we use
    :return: a resized image in rgb
    """

    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image