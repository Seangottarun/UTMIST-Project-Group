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

    # LSTM parameters
    input_dim_lstm = height
    units_in_lstm = 64 
    output_size_lstm = num_classes

    color_channels = 1
    object_labels = ('0', '1', '2', '3', '4',
                     '5', '6', '7', '8', '9')
    num_classes = len(object_labels)
    OutSize=256
    #Wanted to add something
    
    #Emission network parameters
    emi_net_dim = 2
    emi_net_stddev = 0.001    
