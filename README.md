# code_autoencoder_based-is-quanzation-independent
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## target
I complete the whole training and testing system for the starting point to understand data pipeline , autoencoder and quantization. It explores the realation between rate-ditortion and quantization, which is import for learned. compression. 

## usage

### create dataset
first define the your own img_path in create_tfdata.py, and run the create_tfdata.py to get the *.tfrecords.

### training
```
python train_beta.py
```

### testing 
```
python test_beta.py
```
