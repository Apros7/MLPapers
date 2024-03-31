from tinygrad import Tensor
import tinygrad.nn as nn

class Base:
    def __init__(self, layers):
        self.layers = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers): x = layer(x).tanh() if i != len(self.layers) - 1 else layer(x)
        return x

class AutoEncoder:
    def __init__(self, layers):
        self.encoder = Base(layers)
        self.decoder = Base(list(reversed(layers)))
    def __call__(self, x):
        return self.decoder(self.encoder(x))

class AutoEncoderLoss():
    def __init__(self, L1_regularization: float = None, contrastive_regularization: float = None):
        self.l1_lambda = L1_regularization # L1 sparcity regularization
        self.l2_lambda = contrastive_regularization # L2 constrastive 

    
    def __call__(self, x, y, autoencoder: AutoEncoder = None):
        total = ((x - y)**2).sum(axis=1).sum(axis=0)
        l1_lambda = None if autoencoder is None else self.l1_lambda
        total += l1_lambda * 1 if l1_lambda is not None else 0 # still missing something here
        return total