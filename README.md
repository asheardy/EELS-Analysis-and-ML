This repository contains three pieces of code, as described in DOI:

The first "Open and process EELS data.py" allows for loading files from Digital Micrograph, and rapidly processing the data through background subtraction, Fourier-ratio deconvolution, and integrating the pi* and sigma* peaks.

The second "Hybridization Mapping.py" allows for similar processing, and can also bin data to create hybridization maps
  Data for the first two pieces of code can be made available, however the size of the raw files ranges from 60 to 600 MB per file, exceeding the upload limit
The third "CNN Mapping.py" uses EELS data to train and test a convolutional neural network.
  The files "x_train.zip" and "y_train.npy" contain the input training data for the CNN. The files "ML_data_x_test.npy" and "ML_data_y_test.npy" contain the test data.
  
All three have descriptions and comments trying to explain the functions of each step in the code.  These codes generally do not directly output any information, but the use of an environment like Spyder will give outputs of values or plots.
