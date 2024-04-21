from typing import List
import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dims: List[int],
                 output_dim: int
                 ):
        """
        Initialize the MLP with the given parameters

        Args:
            vocab_size (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dims (List[int]): A list of integers representing the sizes of the hidden layers.
            output_dim (int): The size of the output layer.
        """
        super(MultiLayerPerceptron, self).__init__()

        # Save the parameters in case of later use
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Trainable Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size + 1, embedding_dim)

        # Create MLP layers dinamically
        self.hidden_layers = torch.nn.Sequential()
        prev_hidden_dim = embedding_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.hidden_layers.add_module(
                f"linear_{i}", torch.nn.Linear(prev_hidden_dim, hidden_dim))
            self.hidden_layers.add_module(f"relu_{i}", torch.nn.ReLU())
            prev_hidden_dim = hidden_dim

        # Output layer
        self.output_layer = torch.nn.Linear(prev_hidden_dim, output_dim)

        # Sigmoid activation function
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        # Input dims - x: [batch_size, sentence_length]

        x = self.embedding(x)
        # x: [batch_size, sentence_length, embedding_dim]

        x = x.mean(dim=1)
        # x: [batch_size, embedding_dim]

        x = self.hidden_layers(x)
        # x: [batch_size, last_hidden_dim]

        x = self.output_layer(x)
        # x: [batch_size, output_dim]

        # Output dims - x: [batch_size, output_dim]
        return x
