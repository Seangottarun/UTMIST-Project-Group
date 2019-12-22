"""
Parent class for all networks. 

Variables:
config: All variables which are manipulated by the user should be added to the config file.
layers: A call to the Layers() class in baselayer.py, which gives basic layers that can be used to construct the model.
"""
class Network(object):
    def __init__(self, config, layers):
        self.config = config
        self.layers = layers

        


