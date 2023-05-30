from dataclasses import field, dataclass

import numpy as np


@dataclass(slots=True)
class ReLU(object):
    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, x):
        out = np.maximum(0, x)
        self.input = x
        return out

    def backward(self, dout):
        dx = np.array(dout)
        dx[self.input <= 0] = 0
        return dx
