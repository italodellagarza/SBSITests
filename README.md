# ENIAC Tests
Test codes and data for ENIAC 2022 event paper.

[![Python Version](https://img.shields.io/badge/python-3.8.10-green)](https://www.python.org/downloads/release/python-3810/)
[![Virtualenv Version](https://img.shields.io/badge/virtualenv-20.0.17-green)](https://virtualenv.pypa.io/en/20.0.17/user_guide.html)


## Directory Structure
The directory structure is organized by the following:

 - `data/`: Contains all the data used in this project, in Pytorch Geometric data format (`.pt`)
    - `amlsim_31/`: AMLSim 1/3 data files.
    - `amlsim_51/`: AMLSim 5/3 data files.
    - `amlsim_101/`: AMLSim 10/3 data files.
    - `amlsim_201/`: AMLSim 20/3 data files.
 - `models/`: Contains all the code of the models used in this project.
    - `model_gcn.py`: Code for Graph Covolutional Network.
    - `model_skipgcn.py`: Code for Skip-GCN.
    - `model_nenn.py`: Code for NENN
 - `results/`: The results output directory
 - `requirements.txt`: The requirements pip file.
 - `test_gcn_amlsim.py`: Test code for GCN.
 - `test_gcn_xgboost_amlsim.py`: Test code for GCN + XGBoost.
 - `test_skipgcn_amlsim.py`: Test code for Skip-GCN. 
 - `test_skipgcn_xgboost_amlsim.py`: Test code for Skip-GCN + XGBoost. 
 - `test_nenn_amlsim.py`: Test code for NENN. 
 - `test_nenn_xgboost_amlsim.py`: Test code for NENN + XGBoost.

 ## Configure

 You must have Python and virtualenv installed on your system. Once you have it, you must open the project root directory, create a virtual environment, and install the dependencies via pip:

`$ virtualenv -p python3 venv`

`$ source venv/bin/activate`

`$ pip install -r requirements.txt`

## Execute

Once you have configured, you can execute the project. You must have already activated the virtual environment in the root directory. You can execute any test by the following command format:

`$ python <test_to_execute.py> <data_directory> <number_of_repetitions> <output_name>`

For example:

`$ python test_nenn_amlsim.py data/amlsim_31 10 amlsim_31_nenn`

The results will be saved in the `results` directory.