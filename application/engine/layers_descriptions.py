""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
from tensorflow.keras import layers
import keras
from application.engine.preprocessing import ImagePreprocessor
from keras.models import Model

class Architectures:
    """ architectures describe a collection of layers that toghether define a neural netwk
    """
    def __init__(self):
        """   """

class Default(Architectures):
    def __init__(self):
        """ """
        super().__init__()
        self.num_class = 10
        self.inputs = keras.Input(shape=(28, 28))
        # self.outputs = layers.Dense(self.num_class, activation="softmax")
        self.preprocessor = ImagePreprocessor()

    def __call__(self):
        x = self.preprocessor(self.inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(10, activation="softmax")(x)

        model = keras.Model(self.inputs, outputs)
        model.summary()
        return model
