# PiCNet
PiCNet: Physics-infused Convolution Network for Radar-Based Precipitation Nowcasting
The PiCNet is a pytorch-based model for precipitation nowcasting.

For more information or papers, please refer to [PiCNet](https://ieeexplore.ieee.org/document/10890850).

# The short introduction of files

**tool.py**: This file contains some preprocessing function, such as data transfer function, evaluate function, show picture function, etc.

**PiCNet/PiCNet.py**: This file is the kernel file, it builds the whole model, contains the advection simulator module and the physics-guided prediction module and refining network module.

**PiCNet/test.py & train.py**: The former contains the test process of the model. The train.py contains the train process of the model.

# Train
Firstly you should apply for the KNMI dataset, you can apply for the dataset by [KNMI](https://github.com/HansBambel/SmaAt-UNet).

Then, you can use PiCNet/PiCNet/train.py to train your new model or load the pre-trained model.

# Test
You can use PiCNet/PiCNet/test.py to test your model.

# Environment
Python 3.6+, Pytorch 1.0 and Ubuntu.
