"""
Config file to store user-altered variables
"""

class Config():
    # Training Parameters
    lr = 0.01
    num_epochs = 20

    # Dataset Parameters
    batch_size = 100
    height = 28
    width = 28
    color_channels = 1
    object_labels = ('0', '1', '2', '3', '4',
                     '5', '6', '7', '8', '9')
    num_classes = len(object_labels)
        