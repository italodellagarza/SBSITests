#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: 08/05/2022
#
# model_gcn.py: Implementation for GCN GNN architecture
# Based on: https://arxiv.org/pdf/1908.02591.pdf
#

import torch
import random
import torch_geometric.nn as nn

# Seed definitions
torch.manual_seed(2)
random.seed(2)


class GCN(torch.nn.Module):
    """
    Class definition to GCN, with two hidden layers.
    """
    def __init__(self, input_size, hidden_size, class_size):
        """Initialization function

        :param input_size: Size of the node feature size
        :type input_size: int
        :param hidden_size: Size of hidden layers
        :type hidden_size: int
        :param class_size: Class size
        :type class_size: int
        """

        # Superclass initialization
        super(GCN, self).__init__()

        # Define the convolution layers
        self.conv1 = nn.GCNConv(input_size, hidden_size, bias=False)
        self.act1 = torch.nn.ReLU()
        self.conv2 = nn.GCNConv(hidden_size, class_size, bias=False)
        self.act2 = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch_index):
        """Forward function

        :param x: node feature arrays tensor.
        :type x: torch.Tensor
        :param edge_index: node index pairs tensor, defining the edges
        :type edge_index: torch.Tensor
        :param batch_index: tensor defining the batch index (optional).
        :type batch_index: torch.Tensor

        :return: embedding vectors and logits, respectively.
        :rtype: (torch.Tensor, torch.Tensor)
        """
        hidden1 = self.conv1(x, edge_index)
        hidden2 = self.act1(hidden1)
        hidden3 = self.conv2(hidden2, edge_index)
        output = self.act2(hidden3)
        return hidden3, output

