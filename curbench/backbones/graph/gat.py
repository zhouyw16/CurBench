'''
GAT model for node and graph classification from tutorials of torch_geometric
https://github.com/shuowang-ai/graph-neural-network-pyg
https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb
'''


import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GAT4Node(nn.Module):
    def __init__(self, dataset, hidden_channels=8, heads=8, dropout=0.6):
        super(GAT4Node, self).__init__()
        self.gat1 = GATConv(dataset.num_features, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, dataset.num_classes, dropout=dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.6)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x


class GAT4Graph(nn.Module):
    def __init__(self, dataset, hidden_channels=32, heads=8, dropout=0.6):
        super(GAT4Graph, self).__init__()
        self.gat1 = GATConv(dataset.num_features, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.fc = nn.Linear(hidden_channels * heads, dataset.num_classes)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.6)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = self.elu(x)
        x = self.gat2(x, edge_index)
        x = self.elu(x)
        x = self.gat3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x
