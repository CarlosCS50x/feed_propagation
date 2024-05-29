# Neural Network with NumPy

This repository contains a simple implementation of a feedforward neural network using Python and NumPy. The neural network is trained on the XOR dataset to perform binary classification.

## Overview

The neural network architecture consists of an input layer, one hidden layer with four neurons, and an output layer with one neuron. Sigmoid activation function is used for both layers.

## Files

- `neuralnetwork.py`: Contains the implementation of the neural network class (`NeuralNetwork`) along with the sigmoid activation function and its derivative.

## Usage

1. Clone the repository:

    ```
    git clone https://github.com/CarlosCS50x/feed_propagation.git
    ```

2. Navigate to the directory:

    ```
    cd feed_propagation
    ```

3. Run the `neural_network.py` script:

    ```
    python neuralnetwork.py
    ```

4. The script will train the neural network on the XOR dataset for 50,000 iterations and print the final predicted output.

## Dataset

The network is trained on the XOR dataset, which consists of four samples with three features each and corresponding binary labels.

## Requirements

- Python 3.12.3
- NumPy


