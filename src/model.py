from torch_geometric import nn as geom_nn
from torch import nn as torch_nn

class GNNClassifier(torch_nn.Module):
    def __init__(self, in_channels=51, hidden1=102, hidden2=50, hidden3=15, num_classes=8, dropout=0.25):
        super().__init__()

        self.seq = geom_nn.Sequential(
            "x, edge_index",
            [
                (geom_nn.GCNConv(in_channels, hidden1), "x, edge_index -> x"),
                (torch_nn.ReLU(), "x -> x"),
                (torch_nn.Dropout(dropout), "x -> x"),
                (geom_nn.GCNConv(hidden1, hidden2), "x, edge_index -> x"),
                (torch_nn.ReLU(), "x -> x"),
                (torch_nn.Dropout(dropout), "x -> x"),
                (geom_nn.GCNConv(hidden2, hidden3), "x, edge_index -> x"),
                (torch_nn.ReLU(), "x -> x"),
            ]
        )

        self.linear = torch_nn.Linear(hidden3, num_classes)


    def forward(self,  x, edge_index):
        x = self.seq(x, edge_index)
        x = self.linear(x)
        return x
    
