Skintone detection using traditional computer vision algorithms

Came up with various heuristics that could estimate light in the scene and 
output the accurate skin tone of the face.

Tensorflow Installation instructions for M1 Macs:

Tensorflow installed from source since pre-installed binary doesn't work out of the box even with the Rosetta Simulator (x86) running on top of the arm64 instruction set architecture. We have to ensure that the Python interpreter used is 3.8x and not natively installed (since that only works on arm64).

Instructions to install Tensorfow from source on Rosetta Simulator on M1 Mac.

1. Install Homebrew using Rosetta simulator and install Python 3.8 from the website. Related article:  (I don't remember the exact one I used): https://medium.com/thinknum/how-to-install-python-under-rosetta-2-f98c0865e012

2. Run /usr/local/opt/python@3.8/bin/python3 -m venv <directory name> to create a virtual env using the installed Python from Step 1.

3. Download and install Tensorflow from source following: https://www.tensorflow.org/install/source#linux

4. Run "import tensorflow as tf" in the Python console to verify that installation was successful.

4. Move the wheel package from /tmp/tensorflow_pkg to a permanent directory so it can be re-installed for another project (assuming no version update is needed) without having to rebuild from source.


Other dependencies:

1. You will need to include a pre-trained model (not included in this repo) under the model/ directory to perform inference.
2. Install Mask_RCNN private repo to build model to perform inference.
