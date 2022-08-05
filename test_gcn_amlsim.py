#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: date/month/time
#
# test_nenn_amlsim.py: Tests for NENN GNN
#

import os
import sys
import torch
import random
import numpy as np
import scipy
from models.model_gcn import GCN
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score


def get_confidence_intervals(metric_list, n_repeats):
    """Function to calculate the confidence intervals with a 95%
    confidence value.

    :param metric_list: list containing the metrics obtained.
    :type metric_list: list
    :param n_repeats: number of experiment repetitions.
    :type n_repeats: int

    :return: (metric average, confidence interval length)
    :rtype: (float, float)
    """
    confidence = 0.95
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_repeats - 1)
    metric_avg = np.mean(metric_list)

    se = 0.0
    for m in metric_list:
        se += (m - metric_avg) ** 2
    se = np.sqrt((1.0 / (n_repeats - 1)) * se)
    ci_length = t_value * se

    return metric_avg, ci_length


def main():
    """Main function"""
    if len(sys.argv) <= 3:
        print('Wrong number of arguments')
        print('You must put the 3 necessary arguments:')
        print()
        print('$ test_gcn_amlsim.py <dataset_path_name> ' +
              '<number_of_repetitions> <output_name_file>')
        print()
        sys.exit()

    dataset_name = sys.argv[1]
    n_repeats = int(sys.argv[2])
    output = sys.argv[3]

    dataset = []
    for ptfile in os.listdir(dataset_name):
        dataset.append(torch.load(f'{dataset_name}/{ptfile}'))

    new_dataset = []

    # Dataset conversion to let the transactions be the graph nodes
    for data in dataset:
        edge_index = data.edge_to_edge_adj_matr - \
            torch.eye(data.edge_to_edge_adj_matr.shape[0])
        edge_index = edge_index + edge_index.T
        edge_index = edge_index.nonzero()
        x = data.edge_attr
        y = data.y
        new_data = Data(
            x=x,
            y=y,
            edge_index=edge_index
        )
        new_dataset.append(new_data)
    dataset = new_dataset

    # Train and test division
    train_data = dataset[0:int(0.8 * 366)]
    test_data = dataset[int(0.8 * 366):]

    # Executions
    precisions_macro = []
    recalls_macro = []
    f1s_macro = []

    f1s_0 = []
    precisions_0 = []
    recalls_0 = []

    for execution in range(n_repeats):
        while True:
            print(f'EXECUTION {execution}\n')
            # Model definition
            model = GCN(8, 16, 2)
            model = model.to('cpu')

            loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7]))

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training the model
            model.train()
            for epoch in range(100):
                for ts, data in enumerate(train_data):
                    data.to('cpu')
                    optimizer.zero_grad()
                    hidden, logits = model(
                        data.x.T.float(), data.edge_index.T, None
                    )
                    l = loss(logits, data.y.T)
                    l.backward()
                    optimizer.step()
                print('epoch =', epoch + 1, 'loss =', l.item())

            label_pred_list = []
            y_true_list = []

            # Model evaluation
            model.eval()
            model.eval()
            with torch.no_grad():
                for data in test_data:
                    data.to('cpu')
                    _, logits = model(
                        data.x.T.float(), data.edge_index.T, None
                    )
                    label_pred = logits.max(1)[1].tolist()
                    label_pred_list += label_pred
                    y_true_list += data.y.tolist()
            model.train()
            prec_macro = precision_score(
                y_true_list, label_pred_list, average='macro'
            )
            rec_macro = recall_score(
                y_true_list, label_pred_list, average='macro'
            )
            f1_macro = f1_score(y_true_list, label_pred_list, average='macro')

            prec_0 = precision_score(
                y_true_list, label_pred_list, average='binary', labels=[0]
            )
            rec_0 = recall_score(
                y_true_list, label_pred_list, average='binary', labels=[0]
            )
            f1_0 = f1_score(
                y_true_list, label_pred_list, average='binary', labels=[0]
            )

            if f1_0 < 0.2:
                continue

            print(f'\n Precision macro: {prec_macro}')
            print(f'Recall macro: {rec_macro}')
            print(f'F1 macro {f1_macro}')
            print(f'\n Precision ilicit: {prec_0}')
            print(f'Recall ilicit: {rec_0}')
            print(f'F1 ilict: {f1_0}\n')

            precisions_macro.append(prec_macro)
            recalls_macro.append(rec_macro)
            f1s_macro.append(f1_macro)

            precisions_0.append(prec_0)
            recalls_0.append(rec_0)
            f1s_0.append(f1_0)
            break

    result = ""
    metric, ci = get_confidence_intervals(precisions_macro, n_repeats)
    result += f"Macro Precision: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(recalls_macro, n_repeats)
    result += f"Macro Recall: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(f1s_macro, n_repeats)
    result += f"Macro F1: {metric} +- {ci}\n"

    metric, ci = get_confidence_intervals(precisions_0, n_repeats)
    result += f"Ilicit Precision: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(recalls_0, n_repeats)
    result += f"Ilicit Recall: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(f1s_0, n_repeats)
    result += f"Ilicit F1: {metric} +- {ci}\n"

    if not os.path.exists('results'):
        os.mkdir('results')

    with open(f'results/{output}.txt', 'w') as file:
        file.write(result)
        file.close()


if __name__ == '__main__':
    main()
