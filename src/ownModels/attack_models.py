import torch
import torch.nn as nn


class SynonymAttackModelMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_dim, hidden_dims, output_dim):

        super(SynonymAttackModelMLP, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.input_layer = nn.Linear(embedding_dim * input_dim, hidden_dims[0])

        self.attention_layer = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=5,
            padding=2,
        )

        hidden_layers = []
        for i in range(1, len(hidden_dims)):
            hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.hidden_layers = hidden_layers

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.input_layer(x.view(x.size(0), -1))

        x = self.attention_layer(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.relu(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x


class SynonymAttackRNN(nn.Module):
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
        super(SynonymAttackRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = torch.cuda.is_available()

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_layer = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=5,
            padding=2,
        )

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        embeds = self.attention_layer(embeds.permute(0, 2, 1)).permute(0, 2, 1)
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

        if self.train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            )

        return hidden
