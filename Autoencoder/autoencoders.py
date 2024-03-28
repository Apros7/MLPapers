from tinygrad import Tensor
import tinygrad.nn as nn

class Base:
    def __init__(self, layers):
        self.layers = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers): x = layer(x).tanh() if i != len(self.layers) - 1 else layer(x)
        return x

class Encoder(Base):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        pass

class Decoder(Base):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self):
        pass

class AutoEncoder:
    def __init__(self, layers):
        self.encoder = Base(layers)
        self.decoder = Base(list(reversed(layers)))
    def __call__(self, x):
        return self.decoder(self.encoder(x))