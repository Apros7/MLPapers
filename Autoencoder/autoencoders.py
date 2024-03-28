from tinygrad import Tensor
import tinygrad.nn as nn

class Base:
    def __init__(self, input, hidden, output):
        self.layers = [
            nn.Linear(input, hidden),
            nn.Linear(hidden, output)
        ]
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x).tanh() if i != len(self.layers) - 1 else layer(x)
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
    def __init__(self, original_size, hidden_size, compressing_size):
        self.encoder = Base(original_size, hidden_size, compressing_size)
        self.decoder = Base(compressing_size, hidden_size, original_size)
    def __call__(self, x):
        return self.decoder(self.encoder(x))