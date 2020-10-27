""" classes that describe typical models we want to use, e.g. vairous layers architectures
"""
from tensorflow.keras import layers
import keras
from application.engine.preprocessing import ImagePreprocessor
from keras.models import Model
import kerastuner.engine.hyperparameters as hparams
from kerastuner import HyperModel
import yaml
from logger import get_root_logger
from utils import *
LOG = get_root_logger(BASE_LOGGER_NAME)

class Architectures(HyperModel):
    """ architectures describe a collection of layers that toghether define a neural netwk
    """
    def __init__(self, configs, hp_optimize=False, *args, **kwargs):
        """
        :param configs:
        """
        super().__init__(*args, **kwargs)
        self.hp_optimize    = hp_optimize
        self.configs        = configs
        self.mode           = "default"
        self.layer_types    = {"Dense":keras.layers.Dense,
                                "Flatten":keras.layers.Flatten,
                                "Input":keras.Input,
                               "hp.Choice":hparams.Choice,
                               "hp.Int":hparams.Int,
                               "hp.Float": hparams.Float,
                               "hp.Fixed": hparams.Fixed,
                                "hp.Boolean": hparams.Boolean
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

    def build(self, hp):
        """ the tuner expect a build() method it can call on the obj to make the tuner """
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
                # middle_layers.append(self.layer_types[type](units=hp_units, activation=hp_activ))
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return  model

    @property
    def inputs(self):
        return self.configs[self.mode]["architecture"]["inputs"]

class HyperMod(Architectures):
    def __init__(self, configs):
        """ """
        super().__init__(configs)
        self.preprocessor = ImagePreprocessor()

    def __call__(self):
        """ receives the configs loaded from configs.yaml that describe each layers """

        if self.hp_optimize:
            return self._build_model()
        else:
            return self._build_model()
