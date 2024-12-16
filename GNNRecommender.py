import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNNRecommender, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x
