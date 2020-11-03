""" Data access object (DAO).  manages getting stuff from the right place (disk, db, other containers, etc.) """
import os
from utils import *
from logger import get_root_logger
import yaml
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets as datasets
import struct
import numpy as np


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
        self.LOG.info(f"Getting MNIST data.... ")
        train, test = mnist.load_data(path="mnist.npz")
        x_train, y_train = train
        x_test, y_test  = test

        # cutting on the # of samples a bit so things run faster
        x = 4           # reduction factor
        train_s = y_train.shape[0]
        test_s = y_test.shape[0]
        reduced_sample = {"x_train":x_train[:int(train_s/x),:,:], "y_train":y_train[:int(train_s/x)], "x_test":x_test[:int(test_s/x),:,:], "y_test":y_test[:int(test_s/x)]}
        self.LOG.info(f"Got MNIST data(reduced sample): {reduced_sample}")
        return reduced_sample

    def get_yaml_testfile(self):
        """ For test suit """
        with open(os.path.join(TEST_DIR, TEST_FILE), 'r') as configs:
            configs = yaml.load(configs, yaml.FullLoader)
        self.LOG.info(f"Loaded yaml configs: {configs}")
        return configs


    def loadlocal_mnist(images_path, labels_path):
        ########## loaded the code directly from git source, just curious to see how this is done
        # Sebastian Raschka 2014-2020
        # mlxtend Machine Learning Library Extensions
        #
        # A function for fetching the open-source MNIST dataset.
        # Author: Sebastian Raschka <sebastianraschka.com>
        #
        # License: BSD 3 clause

        """ Read MNIST from ubyte files.
        Parameters
        ----------
        images_path : str
            path to the test or train MNIST ubyte file
        labels_path : str
            path to the test or train MNIST class labels file
        Returns
        --------
        images : [n_samples, n_pixels] numpy.array
            Pixel values of the images.
        labels : [n_samples] numpy array
            Target class labels
        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
        """
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                     lbpath.read(8))
            labels = np.fromfile(lbpath,
                                 dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII",
                                                   imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(labels), 784)

        return images, labels


class KerasApiDao(Dao):
    """ responsible for access to data held by the keras api (tensorflow.keras.datasets).

        USAGE: simply define the type & kwargs under architecture.dataset in the yaml config file.

        Then here, we simply map the different types available from Keras (mnist, cifar etc.) to their load_data() method.

        so when we do:

        kapi = KerasApiDao(configs["dataset"])
        data = kapi.get_data()

        we receive the data from whatever is defined in the yaml file.

    """
    def __init__(self, dataset_configs):
        super(KerasApiDao, self).__init__()
        self.configs = dataset_configs
        self.datasource_type_map = {"mnist":datasets.mnist.load_data, "cifar10":datasets.cifar10.load_data}

    def get_data(self):
        train, test = self.datasource_type_map[self.datasource_type](**self.datasource_kwargs)
        x_train, y_train    = train
        x_test, y_test      = test
        data = {"x_train":x_train, "y_train":y_train, "x_test":x_test, "y_test":y_test}
        return data

    def get_data_reduced(self, x=2):
        """ returns as reduced sample by x factor, for faster dev/testing """
        """ just to play around """
        self.LOG.info(f"Getting MNIST data.... ")
        train, test = self.get_data()
        x_train, y_train = train
        x_test, y_test  = test

        # cutting on the # of samples a bit so things run faster
        train_s = y_train.shape[0]
        test_s = y_test.shape[0]
        reduced_sample = {"x_train":x_train[:int(train_s/x),:,:], "y_train":y_train[:int(train_s/x)], "x_test":x_test[:int(test_s/x),:,:], "y_test":y_test[:int(test_s/x)]}
        self.LOG.info(f"Got {self.datasource_type} data(reduced sample): {reduced_sample}")
        return reduced_sample

    @property
    def datasource_type(self):
        return self.configs["type"]

    @property
    def datasource_kwargs(self):
        return self.configs["kwargs"]