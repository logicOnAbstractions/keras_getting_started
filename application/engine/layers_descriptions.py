""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
from tensorflow.keras import layers
import keras
from application.engine.preprocessing import ImagePreprocessor
from keras.models import Model
import yaml

class MyInput:
    def __init__(self, *args, **kwargs):
        self.input = keras.Input(*args, **kwargs)

    def to_yaml(self):
        return self.input.shape
    #
    # def __call__(self, *args, **kwargs):
    #     return self.input(*args, **kwargs)

class Architectures:
    """ architectures describe a collection of layers that toghether define a neural netwk
    """
    def __init__(self,inputs, outputs ):
        """
        :param inputs: keras.Input() layer
        :param outputs: keras.layers.Dense() layer

        the specific architectures should specify a default that makes sense for them. e.g. the MNIST model
        has defaults that fits the standard MNISt data format for example
        """
        self.inputs = inputs
        self.outputs = outputs

    def to_yaml(self):
        """ saves the object (recursively) into yaml repr. we can then re-hydrate that
        yaml into an object, compare/save layers/architectures, etc. """
        d = {}
        for k,v in self.__dict__.items():
            has_todict = getattr(v, "to_dict", None)
            if has_todict:
                d[k] = v.to_dict()
            else:
                d[k] = v


class Default(Architectures):
    # def __init__(self, inputs=keras.Input(shape=(28, 28)), outputs=layers.Dense(10, activation="softmax")):
    def __init__(self, inputs=keras.Input(shape=(28, 28)), outputs=layers.Dense(units=10, activation="softmax")):
        """ """
        super().__init__(inputs, outputs)
        self.num_class = 10
        self.inputs = inputs
        # self.outputs = layers.Dense(self.num_class, activation="softmax")
        self.preprocessor = ImagePreprocessor()
        self.outputs = outputs

    def __call__(self, configs):
        """ receives the configs loaded from configs.yaml that describe each layers """
        x = self.preprocessor(self.inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        model = keras.Model(self.inputs, self.outputs(x))
        model.summary()
        return model
