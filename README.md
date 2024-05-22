# Malaria Cell Detection using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to detect malaria-infected cells from images. The dataset consists of cell images labeled as either parasitized (infected) or uninfected (healthy).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Image Preprocessing](#image-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Malaria is a serious disease caused by parasites, which are transmitted to humans through the bites of infected mosquitoes. Early detection and treatment are crucial to managing the disease. This project leverages deep learning techniques to develop an automated tool for detecting malaria-infected cells.

## Dataset

The dataset used is the **Cell Images Dataset** for malaria detection, which contains labeled images of parasitized and uninfected cells. You can download the dataset from the [Official NIH Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).

### Dataset Structure

- `Parasitized/`: Images of malaria-infected cells.
- `Uninfected/`: Images of healthy cells.

## Model Architecture

The CNN model is designed with the following layers:

- **Convolutional Layers**: Extract features from the input images.
- **Max-Pooling Layers**: Reduce the dimensionality of feature maps.
- **Dense Layers**: Perform the classification based on extracted features.
- **Dropout Layer**: Prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

## Image Preprocessing

The images are preprocessed using the `ImageDataGenerator` class from Keras, which includes:

- Rotation
- Width and height shift
- Shear and zoom
- Horizontal flip

This helps in augmenting the dataset and improving the model's generalization.

## Training

The model is trained with the following settings:

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Epochs**: 20

An early stopping callback is used to prevent overfitting.

## Evaluation

The model is evaluated on a separate test set using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the classification performance.

## Results

The model achieves high accuracy in detecting malaria-infected cells, demonstrating its potential as an effective diagnostic tool.

- **Accuracy**: 94%
- **Precision**: 94%
- **Recall**: 94%
- **F1-Score**: 94%

## Usage

To use this model:

1. Clone the repository.
2. Install the required dependencies.
3. Download and prepare the dataset.
4. Train the model using the provided script.
5. Evaluate the model on the test set.
6. Use the model to predict on new images.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements

- The dataset is provided by the official NIH Malaria Dataset repository.
- This project is inspired by the need for accurate and automated malaria detection tools in medical diagnostics.

For more information, refer to the project's repository: [Malaria Cell Detection](https://github.com/sajidaab89/malaria-cell-detection).
