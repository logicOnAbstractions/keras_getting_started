""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
from tensorflow.keras import layers
import keras
from application.engine.preprocessing import ImagePreprocessor
from keras.models import Model

class Layer:
    def __init__(self):
        """   """

class DefaultLayers(Layer):
    def __init__(self):
        """ """
        super().__init__()
        self.num_class = 10
        self.inputs = keras.Input((28, 28, 1))
        self.outputs = layers.Dense(self.num_class, activation="softmax")
        self.preprocessor = ImagePreprocessor()

    def __call__(self):
        processed_data = self.preprocessor(self.inputs)
        lays        = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(processed_data)
        lays        = layers.MaxPool2D(pool_size=(3,3))(lays)
        lays        = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(lays)
        lays        = layers.MaxPool2D(pool_size=(3,3))(lays)
        lays        = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(lays)
        lays        = layers.GlobalAveragePooling2D()(lays)
        outputs     = self.outputs(lays)
        return keras.Model(inputs=self.inputs, outputs=outputs)
