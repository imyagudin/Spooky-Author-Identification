import torch
from torch import nn

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))

class Model(nn.Module):
    def __init__(self,
                 n_input: int,
                 hidden_size: int,
                 n_output: int
                ) -> None:
        super(Model, self).__init__()
        self.sequential = nn.Sequential(
            nn.Embedding(num_embeddings=n_input, embedding_dim=hidden_size),
            Reorder(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size * 2),
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size * 2, n_output),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        input = self.sequential(input)
        return input