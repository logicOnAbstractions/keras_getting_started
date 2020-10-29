""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
import keras
import tensorflow as tf
from application.engine.preprocessing import ImagePreprocessor
import kerastuner.engine.hyperparameters as hparams
from kerastuner import HyperModel
from logger import get_root_logger
from utils import *
LOG = get_root_logger(BASE_LOGGER_NAME)
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling

class Architectures(HyperModel):
    """ architectures describe a collection of layers that toghether define a neural netwk
    """
    def __init__(self, architecture_configs, *args, **kwargs):
        """
        :param architecture_configs:
        """
        super().__init__(*args, **kwargs)
        self.configs        = architecture_configs
        self.mode           = "default"
        self.layer_types    = {"Dense":keras.layers.Dense,
                                "Flatten":keras.layers.Flatten,
                                "Conv2D": keras.layers.Conv2D,
                                "MaxPooling2D": keras.layers.MaxPool2D,
                               "GlobalAveragePooling2D":keras.layers.GlobalAveragePooling2D,
                                "Input":keras.Input,
                                "CenterCrop": CenterCrop,
                                "Rescaling": Rescaling
                               }
        self.hp_types    = {   "hp.Choice":hparams.Choice,
                               "hp.Int":hparams.Int,
                               "hp.Float": hparams.Float,
                               "hp.Fixed": hparams.Fixed,
                                "hp.Boolean": hparams.Boolean
                               }
        self.preprocessor   = None

    def _build_model(self):
        """ to build a model programatically, we need:
            * a description of an input layer (to match incoming data format)
            * description of the output layers
            * all middle layers descriptions.

            The algo creates input, output layers separately. Just seem convenient for debugging/logic of how a ntwk is structured.
            The middle layers are iterated over

            """
        LOG.info(f"Layers in _buildmodel: {self.configs}")
        inputs_desc     = self.configs["inputs"]
        outputs_desc    = self.configs["outputs"]
        layers_desc     = self.configs["layers"]
        preproc_desc    = self.configs["preprocessor"]

        try:
            type            = inputs_desc['layer_type']
            input_lay       = self.layer_types[type](**{k:v for k,v in inputs_desc.items() if not k=='layer_type'})
            type            = outputs_desc['layer_type']
            output_lay      = self.layer_types[type](**{k:v for k,v in outputs_desc.items() if not k=='layer_type'})
        except Exception as ex:
            LOG.error(f"Failed to build model. input_desc:{inputs_desc} outputdesc: {outputs_desc}")
            LOG.error(f"Layerds: {layers_desc}")
            LOG.error(f"Exception: {ex}")
            exit(-1)

        # now build the preproc/input section first
        layers    = []
        for preproc in preproc_desc:
            layers.append(self._build_layer_from_desc(preproc))

        for layer in layers_desc:
            layers.append(self._build_layer_from_desc(layer))

        # at this point we have all the keras layers obj. in [layers] as per yaml descrip. need to build it up

        # first layer/preproc has particular treatment
        x = layers.pop(0)(input_lay)
        for l in layers:
            print(f"processing layer:{l}")
            x = l(x)
        # all the layers are there - now build the model itself with in, outs
        model = keras.Model(input_lay, output_lay(x))
        return  model

    def build(self, hp):
        """ the tuner expect a build() method it can call on the obj to make the tuner.

            thus this build triggers the tuning phase of the algo
         """
        LOG.info(f"Architecture.build().... configs: {self.configs}")
        inputs_desc     = self.configs["inputs"]
        outputs_desc    = self.configs["outputs"]
        layers_desc     = self.configs["layers"]

        try:
            type            = inputs_desc['layer_type']
            input_lay       = self.layer_types[type](**{k:v for k,v in inputs_desc.items() if not k=='layer_type'})
            type            = outputs_desc['layer_type']
            output_lay      = self.layer_types[type](**{k:v for k,v in outputs_desc.items() if not k=='layer_type'})
        except Exception as ex:
            LOG.error(f"Failed to build model. type: {type}, input_desc:{inputs_desc} outputdesc: {outputs_desc}")
            LOG.error(f"Layerds: {layers_desc}")
            LOG.error(f"exception: {ex}")
        # now iterate on layers
        middle_layers   = []
        for layer in layers_desc:
            type            = layer['layer_type']
            kwargs          = {k:v for k,v in layer.items() if not k == 'layer_type'}

            if "units" in layer.keys() and isinstance(layer["units"], dict):            # then it's a tunable parameters
                units       = layer['units']
                # hp_units = self.hp_types[units["type"]](**units["tunable_kwargs"])
                hp_units = hp.Int(**units["tunable_kwargs"])
                hp_activ = hp.Choice('activation', values=layer['activation']['values'])
                middle_layers.append(self.layer_types[type](units=hp_units, activation=hp_activ))
            else:
                middle_layers.append(self.layer_types[type](**kwargs))

        # we have all layers objs - now build the full model into a single obj.
        x = self.preprocessor(input_lay)
        for l in middle_layers:
            x = l(x)
        # all the layers are there - now build the model itself with in, outs
        model = keras.Model(input_lay ,output_lay(x))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        # model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        #               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=['accuracy'])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      **self.compile_kwargs)
        return  model

    def _build_layer_from_desc(self, layer_desc):
        """ takes in a description, e.g. python, as defined in yaml.configs & returns the layer obj. from keras """
        type            = layer_desc['layer_type']
        kwargs          = {k:v for k,v in layer_desc.items() if not k == 'layer_type'}
        keras_layer = self.layer_types[type](**kwargs)
        return keras_layer

    @property
    def inputs(self):
        return self.configs[self.mode]["architecture"]["inputs"]

    @property
    def compile_kwargs(self):
        """ gets from configs the kwargs to pass to model.compile(), e.g. metrics, losses, etc. we want to use """
        return self.configs["compile"]

class DefaultArch(Architectures):
    def __init__(self, architecture_configs):
        """ """
        super().__init__(architecture_configs)
        LOG.info(f"Instantiating {self.__class__.__name__}")
        self.preprocessor = ImagePreprocessor()

    def __call__(self):
        """ build the model as specified in the configs """

        return self._build_model()


class TunerArch(Architectures):
    """ an architecture meant to optimiz with the kerastuner """