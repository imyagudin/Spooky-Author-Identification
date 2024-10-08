import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 kernel_sizes,
                 output_dim: int,
                 dropout_prob: float = 0.0):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layer = nn.ModuleList([nn.Conv1d(embedding_dim, hidden_dim, kernel_size) for kernel_size in kernel_sizes])
        # self.pooling_layer = nn.ModuleList([nn.AdaptiveMaxPool1d(1), nn.AdaptiveAvgPool1d(1)])
        self.dropout = nn. Dropout(dropout_prob)
        self.full_connected = nn.Sequential(
            # nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim * 3),
            nn.Linear(hidden_dim * 6, hidden_dim * 3),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_dim * 3, hidden_dim * 3)
        )
        self.linear = nn.Linear(hidden_dim * 3, out_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [self.relu(conv(x)) for conv in self.conv_layer]
        # x = [torch.max_pool1d(conv_feat, conv_feat.shape[2]).squeeze(2) for conv_feat in x]
        # x = [nn.MaxPool1d(conv_feat.size(2))(conv_feat).squeeze(2) for conv_feat in x]
        max_pooled = [torch.max(conv_feat, dim=2)[0] for conv_feat in x]
        avg_pooled = [torch.mean(conv_feat, dim=2) for conv_feat in x]
        x = torch.cat(max_pooled + avg_pooled, dim=1)
        # x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.full_connected(x)
        output = self.linear(x)
        return output