# RNA-to-Protein Prediction Using Machine Learning and Deep Learning Models
Project Overview
This project focuses on predicting protein expression from RNA data using a variety of machine learning (ML) and deep learning (DL) models. The models aim to map gene expression levels to corresponding protein levels by training on single-cell multiomics data. The dataset is taken from https://www.kaggle.com/competitions/open-problems-multimodal/overview

## Requirements
python,
numpy,
pandas,
pytables,
h5py,
tables,
pytorch

To run scgpt model https://github.com/bowang-lab/scGPT/tree/f6097112fe5175cd4e221890ed2e2b1815f54010 this needs to be cloned it is the original scgpt model it also cotains the pretrained weights.


## Training
To train the models, run the following commands in the terminal:

Training the Ridge Regression Model:
```
python Ridge_Regression.py
```

Training the Random Forest Model:
```
python Random_Forest.py
```

Training the Fully Connected Neural Network (FCNN):
```
python Neural_network.py
```

Training the Autoencoder-Based Model:
```
python Encoder_Decoder.py
```

Training the SCGPT + Neural Network Model:
```
python rnatoprotein.py
```
