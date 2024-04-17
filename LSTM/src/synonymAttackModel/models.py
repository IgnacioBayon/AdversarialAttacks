import torch
import torch.nn as nn


class SynonymAttackModelMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_dim, hidden_dims):

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.input_layer = nn.Linear(embedding_dim, hidden_dims[0])

        hidden_layers = []
        for i in range(1, len(hidden_dims)):
            hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.hidden_layers = hidden_layers

        self.output_layer = nn.Linear(hidden_dims[-1], input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)

        x = self.ReLU(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.ReLU(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
