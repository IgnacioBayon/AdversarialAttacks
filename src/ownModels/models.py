from typing import List
import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
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
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # Create MLP layers dinamically
        self.hidden_layers = torch.nn.Sequential()
        prev_hidden_dim = embedding_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.hidden_layers.add_module(
                f"linear_{i}", torch.nn.Linear(prev_hidden_dim, hidden_dim)
            )
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
        x = x.long()

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


class SentimentRNN(torch.nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # embedding and LSTM layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )

        # dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = torch.nn.Linear(hidden_dim, output_size)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
        )

        return hidden


class ClassificationRNN(torch.nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        """
        Initialize the model by setting up the layers.
        """
        super(ClassificationRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # embedding and LSTM layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )

        # dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = torch.nn.Linear(hidden_dim, output_size)
        self.sig = torch.nn.Softmax(dim=1)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
        )

        return hidden
