""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
from tensorflow.keras import layers
import keras
from application.engine.preprocessing import ImagePreprocessor
from keras.models import Model
import yaml

class Architectures:
    """ architectures describe a collection of layers that toghether define a neural netwk
    """
    def __init__(self, configs):
        """
        :param inputs: keras.Input() layer
        :param outputs: keras.layers.Dense() layer

        the specific architectures should specify a default that makes sense for them. e.g. the MNIST model
        has defaults that fits the standard MNISt data format for example
        """
        self.configs        = configs
        self.mode           = "default"
        self.layer_types    = {"Dense":keras.layers.Dense,
                                "Flatten":keras.layers.Flatten,
                                "Input":keras.Input}
        self.preprocessor   = None

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

    def _build_model(self):
        inputs_desc     = self.configs["inputs"]
        outputs_desc    = self.configs["outputs"]
        layers_desc     = self.configs["layers"]

        type            = inputs_desc.pop('layer_type', None)
        input_lay       = self.layer_types[type](**inputs_desc)
        type            = outputs_desc.pop('layer_type', None)
        output_lay      = self.layer_types[type](**outputs_desc)

        # now iterate on layers
        middle_layers   = []
        for layer in layers_desc:
            type            = layer.pop('layer_type', None)
            middle_layers.append(self.layer_types[type](**layer))

        # we have all layers objs - now build the full model into a single obj.
        x = self.preprocessor(input_lay)
        for l in middle_layers:
            x = l(x)

        # all the layers are there - now build the model itself with in, outs
        model = keras.Model(input_lay ,output_lay(x))
        return  model

    @property
    def inputs(self):
        return self.configs[self.mode]["architecture"]["inputs"]

class Default(Architectures):
    def __init__(self, configs):
        """ """
        super().__init__(configs)
        self.preprocessor = ImagePreprocessor()

    def __call__(self):
        """ receives the configs loaded from configs.yaml that describe each layers """
        return self._build_model()
