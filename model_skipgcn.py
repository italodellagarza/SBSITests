import torch
import random
import torch_geometric.nn as nn

# Seed definitions
torch.manual_seed(2)
random.seed(2)

class Skip_GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, class_size):
        super(Skip_GCN, self).__init__()
        self.conv1 = nn.GCNConv(input_size, hidden_size, bias=False)
        self.act1 = torch.nn.ReLU()
        self.conv2 = nn.GCNConv(hidden_size, class_size, bias=False)
        #"skip" connection.
        self.W_skip = torch.nn.Linear(input_size, class_size, bias=False)
        self.act2 = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch_index):
        hidden1 = self.conv1(x, edge_index)
        hidden1 = self.act1(hidden1)
        hidden2 = self.conv2(hidden1, edge_index)
        H_skip = self.W_skip(x)
        skip_output = torch.add(hidden2, H_skip)
        output = self.act2(skip_output)

        return hidden2, output