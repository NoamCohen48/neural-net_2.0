from dataclasses import dataclass, InitVar, field

import numpy as np
from numpy.random import Generator


@dataclass(slots=True)
class FullyConnected(object):
    input_size: int = field()
    output_size: int = field()
    # random_generator: Generator = field()

    weights: np.ndarray = field(init=False)
    biases: np.ndarray = field(init=False)

    weights_momentum: np.ndarray = field(init=False)
    biases_momentum: np.ndarray = field(init=False)

    weights_gradient: np.ndarray = field(init=False)
    biases_gradient: np.ndarray = field(init=False)

    input: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        self.weights = np.random.standard_normal((self.input_size, self.output_size)) / np.sqrt(self.input_size / 2)
        self.biases = np.zeros((self.output_size,))
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)

    def forward(self, x, w=None, b=None):
        if w is not None:
            self.W = w
        if b is not None:
            self.b = b
        assert len(x.shape) == 2
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dout):
        self.weights_gradient = np.dot(self.input.T, dout)
        self.biases_gradient = np.sum(dout, 0)
        return np.dot(dout, self.weights.T)

    def update(self, alpha=1e-4, wd=4e-4, momentum=0.9):
        self.weights *= (1 - wd)
        self.biases *= (1 - wd)

        self.weights_momentum = momentum * self.weights_momentum - alpha * self.weights_gradient
        self.biases_momentum = momentum * self.biases_momentum - alpha * self.biases_gradient

        self.weights += self.weights_momentum
        self.biases += self.biases_momentum

        self.weights_gradient = np.zeros_like(self.weights_gradient)
        self.biases_gradient = np.zeros_like(self.biases_gradient)


if __name__ == '__main__':
    from utils.gradient_check import *


    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


    fc = FullyConnected(1024, 10)
    inp = np.random.randn(16, 1024)
    weight = np.random.randn(1024, 10)
    bias = np.random.randn(10, )

    out, cache = fc.forward(inp)
    d_out = np.random.randn(16, 10)
    gradient = fc.backward(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda w: fc.forward(x=inp, w=w)[0], weight, d_out)
    print(rel_error(dw_num, gradient[0]))

    out, cache = fc.forward(inp)
    d_out = np.random.randn(16, 10)
    gradient = fc.backward(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda x: fc.forward(x=inp)[0], inp, d_out)
    print(rel_error(dw_num, gradient[2]))

    out, cache = fc.forward(inp)
    d_out = np.random.randn(16, 10)
    gradient = fc.backward(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda b: fc.forward(x=inp, b=b)[0], bias, d_out)
    print(rel_error(dw_num, gradient[1]))
