import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden
