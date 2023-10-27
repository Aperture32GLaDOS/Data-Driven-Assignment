import numpy as np
from typing import List

class ActivationFunction:
    def forward(self, inputs):
        raise Exception("Do not use the superclass ActivationFunction")

class NeuralLayer:
    def __init__(self, weights, biases, activation_function: ActivationFunction):
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function

    def forward(self, inputs):
        return self.activation_function.forward(np.dot(inputs, self.weights) + self.biases)

class LossFunction:
    def calculate(self, expected, measured):
        raise Exception("Do not use the LossFunction superclass")


class CategoricalCrossEntropy(LossFunction):
    def calculate(self, expected, measured):
        # We are adding a tiny number to expected here to avoid trying np.log(0)
        return -np.sum(measured * np.log(expected + np.array([10**-100])))

class ReLu(ActivationFunction):
    def forward(self, inputs):
        return np.maximum(0, inputs)


class SoftMax(ActivationFunction):
    def forward(self, inputs):
        # You may expect softmax to look like this
        # return np.exp(inputs) / np.sum(np.exp(inputs))
        # Indeed, that is the mathematical definition. However, to defend against overflow, the following defition can be used:
        # return np.exp(inputs - np.max(inputs)) / np.sum(np.exp(inputs - np.max(inputs)))
        # We are subtracting the max of inputs as not too large values can cause overflows (np.exp(710) is the first integer which overflows)
        # But very large negative values do not overflow, as they are approximated to 0. Therefore, it is safe to simply subtract the max of the inputs
        # This is also mathemetically identical to the first definition
        # 
        # For efficiency, we are only calculating the exponentials once
        exps = np.exp(inputs - np.max(inputs))
        return exps / np.sum(exps)



example_layer = NeuralLayer(np.array([1, 1, 1]), np.array([0]), ReLu())
