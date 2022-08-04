#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: date/month/time
#
# model_nenn.py: Implementation for NENN GNN architecture
# Based on: https://proceedings.mlr.press/v129/yang20a.html
#

import torch
import random
import numpy as np
from torch import FloatTensor
from torch.nn import Parameter, LeakyReLU, Module, Softmax, Linear


# Seed definitions
torch.manual_seed(2)
random.seed(2)
np.random.seed(2)


class NodeLevelAttentionLayer(Module):
    """
    Class to Node Level Attention Layer, which calculates an embedding to nodes based on 
    their edges and nodes neighbors.
    """

    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        """
        Initialization function
        :param node_feature_size: int
        :param edge_feature_size: int
        :param node_embed_size: int
        :param edge_embed_size: int
        """

        # Super class initialization
        super(NodeLevelAttentionLayer, self).__init__()

        # Weights definitions
        self.weight_node = Parameter(FloatTensor(node_feature_size, node_embed_size))
        self.weight_edge = Parameter(FloatTensor(edge_feature_size, edge_embed_size))
        self.parameter_vector_node = Parameter(FloatTensor(2 * node_embed_size))
        self.parameter_vector_edge = Parameter(FloatTensor(node_embed_size + edge_embed_size))
        self.importance_normalization = Softmax(dim=0)
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()
        self.edge_embed_size = edge_embed_size
        self.node_embed_size = node_embed_size

        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        torch.nn.init.normal_(self.parameter_vector_node, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.parameter_vector_edge, mean=0.0, std=1.0)

    def forward(self, node_features, edge_features, node_to_node_adj_matrix, edge_to_node_adj_matrix):
        """
        Forward function
        :param node_features: Tensor
        :param edge_features: Tensor
        :param node_to_node_adj_matrix: Tensor
        :param edge_to_node_adj_matrix: Tensor
        """


        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)

        # Node step: calculate node-based neighbors embedding.
        concat_result_n = torch.cat(
            (
                node_embeds.tile([1, node_embeds.shape[0]]).reshape([
                    node_embeds.shape[0], node_embeds.shape[0], node_embeds.shape[1]]),
                node_embeds.tile([node_embeds.shape[0], 1]).reshape(
                    [node_embeds.shape[0], node_embeds.shape[0], node_embeds.shape[1]])
            ),
            dim=2
        )

        attention_output_n = self.node_activation(torch.matmul(concat_result_n, self.parameter_vector_node))

        # Softmax considering only neighbors (non-zero)
        importance_coefficients_n = (
            (
                torch.exp(attention_output_n) / (
                    torch.exp(attention_output_n) *
                    node_to_node_adj_matrix
                ).sum(axis=1)[:, None]
            ) * node_to_node_adj_matrix
        )

        denominator_n = node_to_node_adj_matrix.sum(axis=1)

        embed_propagated_n = node_embeds.tile([node_embeds.shape[0], 1]).reshape(
            [node_embeds.shape[0], node_embeds.shape[0], node_embeds.shape[1]]
        )

        summatory_n = importance_coefficients_n.reshape(
            [importance_coefficients_n.shape[1], importance_coefficients_n.shape[0], 1]
        ) * embed_propagated_n

        final_embeds_n = torch.nan_to_num(summatory_n.sum(axis=1) / denominator_n[:, None])

        output_nodes = self.node_activation(final_embeds_n)

        # Edge step: calculate edge-based neighbors embedding.
        concat_result_e = torch.cat(
            (
                node_embeds.tile([1, edge_embeds.shape[0]]).reshape(
                    [node_embeds.shape[0], edge_embeds.shape[0], node_embeds.shape[1]]),
                edge_embeds.tile([node_embeds.shape[0], 1]).reshape(
                    [node_embeds.shape[0], edge_embeds.shape[0], edge_embeds.shape[1]])
            ),
            dim=2
        )
        attention_output_e = self.edge_activation(torch.matmul(concat_result_e, self.parameter_vector_edge))

        # Softmax considering only neighbors (non-zero)
        importance_coefficients_e = torch.nan_to_num(
            (
                torch.exp(attention_output_e)/(
                    torch.exp(attention_output_e) *
                    edge_to_node_adj_matrix
                ).sum(axis=1)[:, None]
            )*edge_to_node_adj_matrix
        )

        denominator_e = edge_to_node_adj_matrix.sum(axis=1)

        embed_propagated_e = edge_embeds.tile([node_embeds.shape[0], 1]).reshape(
            [node_embeds.shape[0], edge_embeds.shape[0], edge_embeds.shape[1]]
        )

        summatory_e = importance_coefficients_e.reshape(
            [importance_coefficients_e.shape[0], importance_coefficients_e.shape[1], 1]
        ) * embed_propagated_e

        final_embeds_e = torch.nan_to_num(summatory_e.sum(axis=1) / denominator_e[:, None])

        output_edges = self.edge_activation(final_embeds_e)

        return torch.cat((output_nodes, output_edges), dim=1)


class EdgeLevelAttentionLayer(Module):
    """
    Class to Edge Level Attention Layer, which calculates an embedding to edges based on 
    their edges and nodes neighbors.
    """

    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        """
        Initialization function
        :param node_feature_size: int
        :param edge_feature_size: int
        :param node_embed_size: int
        :param edge_embed_size: int
        """

        # Super class initialization
        super(EdgeLevelAttentionLayer, self).__init__()

        # Weights definitions
        self.weight_node = Parameter(FloatTensor(node_feature_size, node_embed_size))
        self.weight_edge = Parameter(FloatTensor(edge_feature_size, edge_embed_size))
        self.parameter_vector_edge = Parameter(FloatTensor(2 * edge_embed_size))
        self.parameter_vector_node = Parameter(FloatTensor(edge_embed_size + node_embed_size))
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()
        self.edge_embed_size = edge_embed_size
        self.node_embed_size = node_embed_size

        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        torch.nn.init.normal_(self.parameter_vector_node, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.parameter_vector_edge, mean=0.0, std=1.0)

    def forward(self, node_features, edge_features, edge_to_edge_adj_matrix, node_to_edge_adj_matrix):
        """
        Forward function
        :param node_features: Tensor
        :param edge_features: Tensor
        :param node_to_node_adj_matrix: Tensor
        :param edge_to_node_adj_matrix: Tensor
        """

        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)

        # Node step: calculate node-based neighbors embedding.
        concat_result_n = torch.cat(
            (
                edge_embeds.tile([1, node_embeds.shape[0]]).reshape(
                    [edge_embeds.shape[0], node_embeds.shape[0], edge_embeds.shape[1]]),
                node_embeds.tile([edge_embeds.shape[0], 1]).reshape(
                    [edge_embeds.shape[0], node_embeds.shape[0], node_embeds.shape[1]])
            ),
            dim=2
        )

        attention_output_n = self.node_activation(torch.matmul(concat_result_n, self.parameter_vector_node))

        # Softmax considering only neighbors (non-zero)
        importance_coefficients_n = torch.nan_to_num(
            (
               torch.exp(attention_output_n) / (
                    torch.exp(attention_output_n) *
                    node_to_edge_adj_matrix
                ).sum(axis=1)[:, None]
            ) * node_to_edge_adj_matrix
        )

        denominator_n = node_to_edge_adj_matrix.sum(axis=1)

        embed_propagated_n = node_embeds.tile([edge_embeds.shape[0], 1]).reshape(
            [edge_embeds.shape[0], node_embeds.shape[0], node_embeds.shape[1]]
        )

        summatory_n = importance_coefficients_n.reshape(
            [importance_coefficients_n.shape[0], importance_coefficients_n.shape[1], 1]
        ) * embed_propagated_n

        final_embeds_n = torch.nan_to_num(summatory_n.sum(axis=1) / denominator_n[:, None])

        output_nodes = self.node_activation(final_embeds_n)

        # Edge step: calculate edge-based neighbors embedding.
        concat_result_e = torch.cat(
            (
                edge_embeds.tile([1, edge_embeds.shape[0]]).reshape(
                    [edge_embeds.shape[0], edge_embeds.shape[0], edge_embeds.shape[1]]),
                edge_embeds.tile([edge_embeds.shape[0], 1]).reshape(
                    [edge_embeds.shape[0], edge_embeds.shape[0], edge_embeds.shape[1]])
            ),
            dim=2
        )

        attention_output_e = self.edge_activation(torch.matmul(concat_result_e, self.parameter_vector_edge))

        # Softmax considering only neighbors (non-zero)
        importance_coeficients_e = torch.nan_to_num(
            (
                torch.exp(attention_output_e)/(
                    torch.exp(attention_output_e) *
                    edge_to_edge_adj_matrix
                ).sum(axis=1)[:, None]
            ) * edge_to_edge_adj_matrix
        )

        denominator_e = edge_to_edge_adj_matrix.sum(axis=1)

        embed_propagated_e = edge_embeds.tile([edge_embeds.shape[0], 1]).reshape(
            [edge_embeds.shape[0], edge_embeds.shape[0], edge_embeds.shape[1]]
        )

        summatory_e = importance_coeficients_e.reshape(
            [importance_coeficients_e.shape[1], importance_coeficients_e.shape[0], 1]
        ) * embed_propagated_e

        final_embeds_e = torch.nan_to_num(summatory_e.sum(axis=1) / denominator_e[:, None])
        output_edges = self.edge_activation(final_embeds_e)

        return torch.cat((output_nodes, output_edges), dim=1)


class Nenn(Module):
    """
    Class definition to NENN, that combines the two types of layers defined above, and
    performs the classificaton via Softmax.
    """

    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size, class_size):
        """
        Initalization function
        :param node_feature_size: int
        :param edge_feature_size: int
        :param node_embed_size: int
        :param edge_embed_size: int
        :param class_size: int
        """

        # Superclass initialization
        super(Nenn, self).__init__()

        # Calculates the intermediate embedding size.
        intermediate_size = node_embed_size + edge_embed_size

        # Layer definitions
        self.layer1 = EdgeLevelAttentionLayer(node_feature_size, edge_feature_size, node_embed_size, edge_embed_size)
        self.layer2 = NodeLevelAttentionLayer(node_feature_size, intermediate_size, node_embed_size, edge_embed_size)
        self.layer3 = EdgeLevelAttentionLayer(intermediate_size, intermediate_size, node_embed_size, edge_embed_size)
        self.layer4 = Linear(intermediate_size, class_size)
        self.layer5 = Softmax(dim=1)

    def forward(
            self,
            node_features,
            edge_features,
            edge_to_edge_adj_matrix,
            edge_to_node_adj_matrix,
            node_to_edge_adj_matrix,
            node_to_node_adj_matrix,
    ):
        """
        Foward function:
        :param node_features: Tensor
        :param edge_features: Tensor
        :param edge_to_edge_adj_matrix: Tensor
        :param node_to_edge_adj_matrix: Tensor
        :param node_to_node_adj_matrix: Tensor
        """
        edge_embeds1 = self.layer1(node_features, edge_features, edge_to_edge_adj_matrix, node_to_edge_adj_matrix)
        node_embeds = self.layer2(node_features, edge_embeds1, node_to_node_adj_matrix, edge_to_node_adj_matrix)
        edge_embeds2 = self.layer3(node_embeds, edge_embeds1, edge_to_edge_adj_matrix, node_to_edge_adj_matrix)
        output_linear = self.layer4(edge_embeds2.float())
        output = self.layer5(output_linear)
        return output, edge_embeds2

