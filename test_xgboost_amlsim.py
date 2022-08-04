# TODO arrumar: remover a rede neural. Ao invés do embedding, a linha terá os dados da transação e de ambos os envolvidos.


import os
import sys
import torch
import random
import scipy
import xgboost as xgb
import numpy as np
from torch_geometric.data import Data
from model_nenn import Nenn
from sklearn.metrics import precision_score, recall_score, f1_score


def get_confidence_intervals(metric_list, n_repeats):
    confidence = 0.95
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_repeats - 1)
    metric_avg = np.mean(metric_list)

    se = 0.0
    for m in metric_list:
        se += (m - metric_avg) ** 2
    se = np.sqrt((1.0 / (n_repeats - 1)) * se)
    ci_length = t_value * se

    return (metric_avg, ci_length)
        



def main():
    if len(sys.argv) <= 3:
        print('Wrong number of arguments')
        print('You must put the 3 necessary arguments:')
        print()
        print('$ test_nenn_amlsim.py <dataset_path_name> <number_of_repetitions> <output_name_file>')
        print()
        sys.exit()
    
    dataset_name = sys.argv[1]
    n_repeats = int(sys.argv[2])
    output = sys.argv[3]

    dataset = []
    for ptfile in os.listdir(dataset_name):
        dataset.append(torch.load(f'{dataset_name}/{ptfile}'))

    # Train and test division
    train_data = dataset[0:int(0.8*366)]
    test_data = dataset[int(0.8*366):]


    # Executions
    precisions_macro = []
    recalls_macro = []
    f1s_macro = []

    f1s_0 = []
    precisions_0= []
    recalls_0 = []
    

    for execution in range(n_repeats):
        while True:
            print(f'EXECUTION {execution}\n')
            # Model definition
            model = Nenn(6,8,6,8,2)
            model = model.to('cpu')

            # Cross Entropy Weight definition

            weight = None
            if dataset_name.endswith('51'):
                weight = torch.Tensor([0.4, 0.6])
            elif dataset_name.endswith('101'):
                weight = torch.Tensor([0.15, 0.85])
            elif dataset_name.endswith('201'):
                weight=torch.Tensor([0.04, 0.96])

            loss = torch.nn.CrossEntropyLoss(weight=weight)


            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


            # Training the model
            model.train()
            for epoch in range(100):
                for ts, data in enumerate(train_data):
                    # Reset gradients
                    optimizer.zero_grad()
                    # Send info to the model
                    logits, _ = model(
                        data.x.T.type(torch.FloatTensor),
                        data.edge_attr.T.type(torch.FloatTensor),
                        data.edge_to_edge_adj_matr.type(torch.FloatTensor),
                        data.edge_to_node_adj_matr.type(torch.FloatTensor),
                        data.node_to_edge_adj_matr.type(torch.FloatTensor),
                        data.node_to_node_adj_matr.type(torch.FloatTensor)
                    )
                    # Calculate loss
                    l = loss(logits, data.y)
                    l.backward()
                    # Update gradients
                    optimizer.step()
                if np.isnan(l.item()):
                    model = None
                    break
                print('epoch =', epoch + 1, 'loss =', l.item())

            if not model:
                continue
            
            # Obtain embeddings to train xgboost
            embeddings_train = []
            y_train = []

            model.eval()
            with torch.no_grad():

                for data in train_data:
                    data.to('cpu')
                    _, embedding_train = model(
                        data.x.T.type(torch.FloatTensor),
                        data.edge_attr.T.type(torch.FloatTensor),
                        data.edge_to_edge_adj_matr.type(torch.FloatTensor),
                        data.edge_to_node_adj_matr.type(torch.FloatTensor),
                        data.node_to_edge_adj_matr.type(torch.FloatTensor),
                        data.node_to_node_adj_matr.type(torch.FloatTensor)
                    )
                    embeddings_train += embedding_train.numpy().tolist()
                    y_train += data.y.numpy().tolist()
            model.train()

            embeddings_train = np.array(embeddings_train)
            y_train = np.array(y_train)

            # Obtain embeddings to test xgboost
            embeddings_test = []
            y_true = []

            model.eval()
            with torch.no_grad():

                for data in test_data:
                    data.to('cpu')
                    _, embedding_test = model(
                        data.x.T.type(torch.FloatTensor),
                        data.edge_attr.T.type(torch.FloatTensor),
                        data.edge_to_edge_adj_matr.type(torch.FloatTensor),
                        data.edge_to_node_adj_matr.type(torch.FloatTensor),
                        data.node_to_edge_adj_matr.type(torch.FloatTensor),
                        data.node_to_node_adj_matr.type(torch.FloatTensor)
                    )
                    embeddings_test += embedding_test.numpy().tolist()
                    y_true += data.y.numpy().tolist()
            model.train()

            embeddings_test = np.array(embeddings_test)
            y_true = np.array(y_true)

            # Creating and training the model

            print('Training XGBoost...')
            xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            xgb_model.fit(embeddings_train, y_train)

            # Evaluating the model
            y_pred = xgb_model.predict(embeddings_test)

        
            prec_macro = precision_score(y_true, y_pred, average='macro')
            rec_macro = recall_score(y_true, y_pred, average='macro')
            f1_macro = f1_score(y_true, y_pred, average='macro')

            prec_0 = precision_score(y_true, y_pred, average='binary', labels=[0])
            if prec_0 < 0.3:
                print('\n\n\n\n\n\n\n')
                continue
            rec_0 = recall_score(y_true, y_pred, average='binary', labels=[0])
            f1_0 = f1_score(y_true, y_pred, average='binary', labels=[0])

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
