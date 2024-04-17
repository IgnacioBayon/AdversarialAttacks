from typing import List, Dict, Tuple
import torch
import numpy as np
import random
import os


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, vocab_size: int, sentence_length: int, embedding_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        """
        Initialize the MLP with the given parameters

        Args:
            vocab_size (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dims (List[int]): A list of integers representing the sizes of the hidden layers.
            output_dim (int): The size of the output layer.
        """
        super(MultiLayerPerceptron, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Trainable Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size + 1, embedding_dim)

        # Create MLP layers dinamically
        self.hidden_layers = torch.nn.Sequential()
        # self.hidden_layers.add_module("flatten", torch.nn.Flatten())
        prev_hidden_dim = embedding_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.hidden_layers.add_module(
                f"linear_{i}", torch.nn.Linear(prev_hidden_dim, hidden_dim))
            self.hidden_layers.add_module(f"relu_{i}", torch.nn.ReLU())
            # self.hidden_layers.add_module(
            #     f"dropout_{i}", torch.nn.Dropout(dropout))
            prev_hidden_dim = hidden_dim

        # Output layer
        self.output_layer = torch.nn.Linear(prev_hidden_dim, output_dim)

        # Global Average pooling
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        # # Sigmoid activation function
        # self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        # Initial input dims - x: [batch_size, sentence_length]

        x = self.embedding(x)
        # x: [batch_size, sentence_length, embedding_dim]

        x = x.mean(dim=1)
        # x: [batch_size, embedding_dim]

        x = self.hidden_layers(x)
        # x: [batch_size, last_hidden_dim]

        x = self.output_layer(x)
        # x: [batch_size, output_dim]

        x = x.squeeze()
        # x: [batch_size]

        return x
