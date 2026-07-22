# MNIST Classification: Network Architecture Design

## Objective
Improve MNIST classification accuracy by designing a better neural network architecture. Your code goes in the `Net` class (lines 10-33) in `main.py`.

## Background
The training pipeline is fixed: 14 epochs, Adadelta optimizer, StepLR scheduling, batch size 64, NLL loss on log-softmax output. The default Net uses two conv layers + two FC layers, achieving ~99.2% accuracy.
