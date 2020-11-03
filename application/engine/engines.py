""" I still have to fully clarify for myself what the Engine is, and isn't.

    But broadly:
        - the engine includes the model
        - however it may perform additional stuff on top of the model
        - it unifies the tuner, training & predictions into a single class hierarchy
        - eventually, would be responsible to load/save a model that's already been trained etc.
        - figure out what dataset to use that sort of things
 """

from kerastuner.tuners import RandomSearch
import keras
import kerastuner.engine.hyperparameters as hp
from application.engine.layers_descriptions import DefaultArch
from dao import DiskDao, KerasApiDao
from utils import *
from logger import get_root_logger

LOG = get_root_logger(BASE_LOGGER_NAME)

class Engine:
    """ parent class, meant to be subclasses by specific implt. models
        will allow flexibility in using various models to make predictions
    """
    def __init__(self, architecture_configs):
        """ """
        LOG.info(f"Instantiating {self.__class__.__name__}")
        self.dao            = DiskDao()
        self.configs        = architecture_configs
        self.mode           = "default"         # TODO: for now
        self.datasource     = None
        self.preprocessor   = None
        self.model          = DefaultArch(architecture_configs)
        self.datasource_map = {"KerasApiDao":KerasApiDao, }         # just one for now, may add others

        self.init_components()

    def init_components(self):
        """ instantiate any obj that makes sense based on the configs """
        self.datasource = self.datasource_map[self.datasource_str](self.dataset_confs)

    def execute(self):
        """ defines what the model is suppose to perform as action """
        raise NotImplemented

    def predict_single(self, data):
        """ makes a prediction. subclasses define what is predicted """
        raise NotImplemented

    def load_model(self):
        """ Loads the appropriate model from keras """
        raise NotImplemented
    def instantiante_layers(self):
        """ uses content of self.configs to instantiate all of our model. this uses the functional notation, which is fine
         for non-tuners model, e.g. the default (see Tuner.instantiate_layers() override for details). """
        self.model = self.model()
        self.model.summary()

    @property
    def compile_kwargs(self):
        """ gets from configs the kwargs to pass to model.compile(), e.g. metrics, losses, etc. we want to use """
        return self.configs["compile"]

    @property
    def datasource_str(self):
        return self.dataset_confs["datasource"]

    @property
    def dataset_confs(self):
        """ where to get the data for this model """
        return self.configs["dataset"]

    @property
    def data_reduced(self):
        """ returns data as per self.datasource, e.g. the source we have specified in the yaml configs """
        return self.datasource.get_data_reduced(x=10)

    @property
    def data(self):
        return self.datasource.get_data()


class PredictionEngine(Engine):
    """ sample model that trains to recognize the MNIST digits classic example """

    def __init__(self, architecture_configs):
        """ """
        super().__init__(architecture_configs)
        LOG.info(f"Instantiating {self.__class__.__name__}")
        self.instantiante_layers()

    def execute(self):
        self.train()

    def train(self):
        """ launches all the steps necessary to preprocess data, make predictions, etc. """

        # proprocess the data
        data = self.data
        LOG.info(f"got data from keras: {data.keys()}")
        # pass it to our model - the model also takes care of preprocessing so we just pass it the raw data we loaded
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        self.model.fit(data["x_train"],data["y_train"])
        LOG.info(f"Finished training model. {self.model.history}")

    def predict_single(self, data):
        """ makes a prediction, assuming a trained model. if not will return random answer """


class TunerEngine(Engine):
    """
        currently we test for the keras tuner thingy, so this models makes default choiecs
        to make that happen in the layer choices etc.
     """
    def __init__(self, architecture_configs):
        """ """
        super().__init__(architecture_configs)
        LOG.info(f"Instantiating {self.__class__.__name__}")

    def execute(self):
        """ Specifies what happens when we want to launch this model."""
        self.tune()

    def tune(self):
        """ TODO: actually this should be def tune(..) - will have to reoncile/fix the nomentalures at some point """
        tuner = RandomSearch(self.model, objective='val_accuracy', max_trials=50, executions_per_trial=2,
                             directory='my_dir', project_name='helloworld')

        # proprocess the data
        data = self.data_reduced
        LOG.info(f"BUILD: data from keras: {data.keys()}")

        x = data["x_train"]
        y = data["y_train"]
        x_val = data["x_test"]
        y_val = data["y_test"]

        print(f"label shapes: {y.shape} {y_val.shape}")
        # call the tuner on that

        try:
            tuner.search(x, y, epochs=3, validation_data=(x_val, y_val))
        except Exception as ex:
            LOG.error(f"Failed to tuner.search: {ex}")
        models = tuner.get_best_models(num_models=2)

        # self.tuner_model.fit(data["x_train"],data["y_train"])
        LOG.info(f"Finished tuning model. Summary:")

    def predict_single(self, data):
        """ makes a prediction, assuming a trained model. if not will return random answer """

    def instantiante_layers(self):
        """ known issue: cannot build (e.g. call the functional keras.Layer()() form ) right away.
            the kerastuner expects the model to build only at runtime, after we pass it kerastuner.engines.hyperparameters (hp))
        """
        self.model = DefaultArch(self.configs)        # NOTE: we DONT use the functional notation here, because the actual model needs to be built at runtime when self.tuner.build(hp) is called by the kerastuner
        self.model.summary()
