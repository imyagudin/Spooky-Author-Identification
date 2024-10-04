import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_classes: int,
                 kernel_sizes,
                 num_filters: int):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layer = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        self.relu = nn.ReLU()
        self.pooling_layer = nn.ModuleList([nn.AdaptiveMaxPool1d(1), nn.AdaptiveAvgPool1d(1)])
        self.dropout = nn. Dropout(0.2)
        self.full_connected = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_filters * 3, num_filters * 3)
        )
        self.linear = nn.Linear(num_filters * 3, out_features=num_classes)

    def forward(self, input):
        input = self.embedding(input)
        input = input.permute(0, 2, 1)
        input = [torch.relu(conv(input)) for conv in self.conv_layer]
        input = [torch.max_pool1d(conv_feat, conv_feat.shape[2]).squeeze(2) for conv_feat in input]
        input = torch.cat(input, dim=1)
        output = self.linear(input)
        return output