"""" .py contenant les tests à effectuer"""

import unittest                         # our test lib
from logger import get_root_logger
from utils import *
from classes.arg_parser import ArgParser
from dao import DiskDao
from application.engine.engines import DigitsMNIST, Tuner

class SomeTests(unittest.TestCase):     # on doit hériter de TestCase

    def __init__(self, *args, **kwargs):
        super(SomeTests, self).__init__(*args, **kwargs)
        self.main_prog = None
        self.dao        = DiskDao()
        self.confs      = self.dao.get_yaml_testfile()

    def setUp(self):
        """ Runs after EACH test. Here we instantiate a new instance
         each test because we don't want the values modified by a previous test to influence the results of the next one"""

    def tearDown(self):
        """
        Runs after each test. Similar to class teardown
        """
        self.main_prog = None

    ################################################ tests

    def test_yaml_loader(self):
        """ tests that we receive proper yaml when loading """
        test_confs = self.dao.get_yaml_testfile()["tests"]
        self.assertTrue(isinstance(test_confs, dict))

        # accessing a few keys that should be present
        for key in ["main_program",]:
            self.assertTrue(key in test_confs.keys())

    def test_regular_model_MNIST(self):
        """ Loads a model/a few described in the tests configs & runs them to check they're valid
            Using our DigitMNIST class for now
         """
        for k,v in self.archs.items():
            LOG.info(f"Testing architecture {k}")
            model = DigitsMNIST(configs=v)
            model.train()

    def test_tuner_model_MNIST(self):
        """ Loads a model/a few described in the tests configs & runs them to check they're valid
            Using our DigitMNIST class for now
         """
        for k,v in self.archs.items():
            LOG.info(f"Testing architecture {k}")
            model = Tuner(configs=v)
            model.train()


    @property
    def archs(self):
        return self.confs["tests"]["architectures"]

    @property
    def tuner_archs(self):
        return self.confs["tests"]["tuner_archs"]

if __name__ == '__main__':
    import sys
    LOG = get_root_logger(BASE_LOGGER_NAME, filename=f'tests.log')
    LOG.debug(f'logger debug level msg ')
    LOG.info(f'logger info level msg ')
    LOG.warning(f'logger warn level msg ')
    LOG.error(f'logger error level msg ')
    LOG.critical(f'logger critical level msg ')

    argparser = ArgParser(logger=LOG)
    # we've parsed the args, so remove them so it doesn't trip up unitttest when it doesn't recognize them itself
    sys.argv[1:] = []
    unittest.main()