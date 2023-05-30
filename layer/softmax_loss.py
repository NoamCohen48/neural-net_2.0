from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class Softmax_and_Loss(object):
    pred: np.ndarray | None = field(init=False, default=None)
    def forward_and_backward(self, x, y):
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.pred = probs
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y - 1])) / N
        dx = probs.copy()
        dx[np.arange(N), y - 1] -= 1.0

        dx /= N  # for the loss is divided by N
        return loss, dx

    def backward(self):
        pass


if __name__ == '__main__':
    from utils.gradient_check import *


    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


    inp = np.random.randn(32, 10)
    target = np.ones(32, dtype=int)
    dout = np.ones_like(inp)

    softmax = Softmax_and_Loss()
    gradient = softmax.forward_and_backward(inp, target)[1]
    dx_num = eval_numerical_gradient_array(lambda x: softmax.forward_and_backward(x, target)[0], inp, 1)

    print(rel_error(dx_num, gradient))
