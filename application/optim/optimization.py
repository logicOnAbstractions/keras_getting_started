""" contains the different optimizers we can use to refine our learning models """
""" a few notes not obvious from the tuto/get started:

    * hp is a module that contains the variable hyperparameters options for model tuning.
        * options: hp.Choice, hp.Int, hp.Boolean, hp.Float, hp.Fixed

"""

from kerastuner import HyperModel as HypMod
import kerastuner.engine.hyperparameters as hp


class KerasTuner(HypMod):
    def __init__(self, model=None, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        self.model      = model

    def build(self, hp):
        """ """

        # we already have the model

        # however that model have fixed values, in its layer description, which we need to replace by our hp.Choice, hp.Int etc.

        # that means we either need to go through updating all the layer's hp,
        # or perhaps better yet have our model be able to return a usable form (default)
        # or as optimizable form, where we specify which param should be regular int/float/etc.
        # and which ones need to be hp.Stuff(...)
