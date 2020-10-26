""" some data preprocessing options, taken from the keras docs """
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, Normalization, CenterCrop, Rescaling


class Preprocessor:
    """ defines generic behavior of a preprocessor """
    def __init__(self, data=None, *args, **kwargs):
        """ """
        self.processor          = None
        if data:
            self.adapt(data)

    def adapt(self, data):
        """ generates a vocabulary index. not 100% sure what this means.
            seems to be a kind of index created based on the data
         """
        self.processor.adapt(data)

    def encode(self, data):
        """ takes in data in list format & returns its encoded form  """
        raise NotImplementedError


class TextPreprocessor(Preprocessor):
    """ defines generic behavior of a preprocessor """
    def __init__(self, data=None, output_mode="int", ngrams=2):
        """ """
        super().__init__(data=data)
        self.output_mode        = output_mode
        self.ngrams             = ngrams
        self.processor          = TextVectorization(output_mode=output_mode, ngrams=ngrams)

    def encode(self, data):
        """ takes in data in list format & returns its encoded form  """
        return self.processor(data)


class NormalizePreprocessor(Preprocessor):

    def __init__(self, data=None, axis=-1):
        """ typically the last axis is the one we normalize over """
        super().__init__(data=data)
        self.processor          = Normalization(axis=axis)

    def encode(self, data):
        """ takes in data in list format & returns its encoded form  """
        return self.processor(data)


class ImagePreprocessor(Preprocessor):

    def __init__(self):
        """ typically the last axis is the one we normalize over """
        super().__init__()
        # self.cropper            = CenterCrop(height=height, width=width)
        # self.scaler             = Rescaling(scale=scale)

    def __call__(self, inputs):
        """ takes in data in list format & returns its encoded form  """
        x = Rescaling(scale=1.0/255)(inputs)
        return x