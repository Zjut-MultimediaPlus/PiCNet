# PiCNet
PiCNet: Physics-infused Convolution Network for Radar-Based Precipitation Nowcasting
The PiCNet is a pytorch-based model for precipitation nowcasting.

For more information or papers, please refer to [PiCNet](https://ieeexplore.ieee.org/document/10890850).

# The short introduction of files

**tool.py**: This file contains some preprocessing function, such as data transfer function, evaluate function, show picture function, etc.

**PiCNet/PiCNet.py**: This file is the kernel file, it builds the whole model, contains the advection simulator module and the physics-guided prediction module and refining network module.

**Adveciton/test.py & train.py**： The former contains the test process of the Advection smutilator. The train.py contains the training process of the Advection smutilator.

**PiCNet/test.py & train.py**: The former contains the test process of the model. The train.py contains the training process of the model.

# Train
Firstly you should apply for the KNMI dataset, you can apply for the dataset by [KNMI](https://github.com/HansBambel/SmaAt-UNet).

Then, you can use PiCNet/Adveciton/train.py and PiCNet/PiCNet/train.py to train your new model.

The results of the Advection smutilator are trained by PiCNet/Adveciton/train.py.

# Test
You can use PiCNet/PiCNet/test.py to test your model.

# Environment
Python 3.6+, Pytorch 1.0 and Ubuntu.
