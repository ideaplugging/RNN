import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            n_layers=4,
            dropout_p = .2,
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p, # basic
            bidirectional=True,
        )

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (bs, h, w) - gray scale / h = time step, w = magnitude of the vectors

        z, _ = self.rnn(x) # input의 hidden과 cell state는 명시하지 않을 경우 0으로 초기화
        # |z| = (bs, height, hidden_size * 2) - 전체 time step의 것
        z = z[:, -1] # 마지막 것만 가지고 올 것
        # |z| = (bs, hs * 2)
        y = self.layers(z)
        # |y| = (bs, os)

        return y
