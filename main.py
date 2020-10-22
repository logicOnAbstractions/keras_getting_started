""" A generic main program to wrap around any application. Loads yaml configs, logs, & other nice defaults thigns.
"""

from utils import *
from classes.arg_parser import ArgParser
from logger import get_root_logger
from dao import DiskDao
from application.engine.engines import DogBreedModel, DigitsMNIST

class MainProgram:
    """ the runner of the simulation
        TODO: add a reset to return to start/default state of some kind
     """
    def __init__(self, arg_parser=None, logger=None):
        # those arguments may be set from the config file
        self.foo            = None

        self.model          = DigitsMNIST()
        self.dao            = DiskDao()
        self.LOG            = logger if logger else self.get_logger()
        self.arg_parser     = arg_parser if arg_parser else ArgParser(self.LOG)
        self.init_args()

        self.start_webserver()

        self.LOG.info(f"Done init in {self.__class__.__name__}.")

        # auto start
        self.execute_all()

    def init_args(self):
        """ parses whatever args we have & sets up this class accordingly

            attribute must already be declared in the class - otherwise they are ignored
        """
        self.arg_parser.parse_cmdline()
        self.full_configs = self.arg_parser.parse_yaml_configs()
        for k, v in self.configs_default_mp.items():
            if hasattr(self, k):
                setattr(self, k, v)
                self.LOG.info(f"Attribute {k} has been set to {getattr(self, k)}")
            else:
                self.LOG.warning(f"Config file contains an attribute that is not in this class's attribute and therefore has not been set (k,v): {k}, {v}")

        if self.mode != "default":                  # then those will complement/override the default values
            for k, v in self.full_configs[self.mode]["main_program"].items():
                if hasattr(self, k):
                    setattr(self, k, v)
                    self.LOG.info(f"Attribute {k} has been set to {getattr(self, k)}, overriding default values if any.")
                else:
                    self.LOG.warning(f"Config file contains an attribute that is not in this class's attribute and therefore has not been set (k,v): {k}, {v}")

    def start_webserver(self):
        """ starts the webserver for the flask api """
        self.LOG.info(f"Starting webserver... (not implemented)")

    def predict(self, data):
        """ abstration layer. we feed that data to the model we have set to make a prediction """
        return self.model.predict(data)

    def execute_all(self):
        return self.model.excute_all()

    @property
    def configs_default_mp(self):
        """ returns the part of the config file that contains the configs by default for main program """
        return self.full_configs["default"]["main_program"]

    @property
    def configs(self):
        return self.dao.get_configs_yaml()

    @property
    def mode(self):
        """ the config mode we want. defaults is set by argparser at default"""
        return self.args.mode

    @property
    def args(self):
        return self.arg_parser.args

    def get_logger(self):
        """ inits the logs. should only be if for whatever reason no logger has been defined """
        logger = get_root_logger(BASE_LOGGER_NAME, filename=f'log.log')
        logger.info(f"Initated logger in {self.__class__.__name__} ")
        logger.debug(f'logger debug level msg ')
        logger.info(f'logger info level msg ')
        logger.warning(f'logger warn level msg ')
        logger.error(f'logger error level msg ')
        logger.critical(f'logger critical level msg ')
        return logger

if __name__ == '__main__':
    mp = MainProgram()
