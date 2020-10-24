import argparse
import yaml
from utils import *
from logger import get_root_logger
from dao import DiskDao


class ArgParser:
    """ contains the logic to get arguments from the cmd line & passes them on to the program"""
    def __init__(self, logger=None):
        self.LOG = logger if logger else get_root_logger(BASE_LOGGER_NAME)
        self.LOG.info(f"Parsing args in {self.__class__.__name__}")
        self.configs_file       = None
        self.dao                = DiskDao()

    def parse_cmdline(self):
        """ parses stuff.

            to use the values from destinations:
            parser.add_argument('--grid-size', dest='N')
            self.args = parser.parse_args()     # assigns everything to mapping
            (... code.... )

            print(self.args.N)                  # accessing value of --grid-size store as variable N
        """
        # parse arguments
        self.LOG.info(f"ArgParser - parsing commandline args... ")
        parser = argparse.ArgumentParser(description="YOUR PROGRAM NAME")

        # assign args to vars ("dest=...")
        parser.add_argument('--configsfile', dest='configs_file', required=False, type=str, default="configs.yaml")
        parser.add_argument('--configsmode', dest='mode', required=False, type=str, default="default")                 # just use default configs by default

        # awesome. we now get a mapping of args according to what we wrote above
        self.args = parser.parse_args()
        self.configs_file = self.args.configs_file
        self.LOG.info(f"Args from cmdline: {self.args}")

    def parse_yaml_configs(self):
        """ reads a properly formatted yaml configuration file & parses it into python dictionary format  """
        return self.dao.get_configs_yaml()
