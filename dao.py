""" Data access object (DAO). manages getting stuff from the right place (disk, db, other containers, etc.) """
import os
from utils import *
from logger import get_root_logger
import json
import yaml
import tensorflow.keras.datasets.mnist as mnist


class Dao:
    """ super class for accessors (file, nwtk etc.)"""
    def __init__(self):
        self.LOG = get_root_logger(BASE_LOGGER_NAME)

class DiskDao(Dao):
    def __init__(self):
        super().__init__()

    # TODO: wrap in try-catch. Better - decorators to wrap those in try-catch
    def get_configs_yaml(self):
        """ reads a properly formatted yaml configuration file & parses it into python dictionary format  """
        with open(CONFIGS_FILE, 'r') as configs:
            configs = yaml.load(configs, yaml.FullLoader)
        self.LOG.info(f"Loaded yaml configs: {configs}")
        return configs

    def get_mnist_dataset(self):
        """ just to play around """
        train, test = mnist.load_data(path="mnist.npz")
        x_train, y_train = train
        x_test, y_test  = test
        return {"x_train":x_train, "y_train":y_train, "x_test":x_test, "y_test":y_test}
