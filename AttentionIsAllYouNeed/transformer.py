import torch

N_encoder_layers = 6
N_decoder_layers = 6
N_encoder_heads = 8
N_decoder_heads = 8
d_model = 512
d_ff = 2048

# multi-head attention
h = 8
d_k = d_model/h
d_v = d_model/h

def attention(q, k, v): # dot scaling + 1/d_model**0.5 scaling
    return torch.softmax(q @ k.T / (d_model ** 0.5), dim=-1) @ v

class Transformer:
    def __init__(self) -> None:
        pass

class TransformerBlock:
    def __init__(self) -> None:
        pass

class Attentionblock:
    def __init__(self) -> None:
        pass