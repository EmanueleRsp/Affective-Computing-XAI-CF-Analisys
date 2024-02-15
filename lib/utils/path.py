"""Define directories and paths managed in the project."""

import os
import sys

METHODS = ['No normalization', 'MinMaxScaler', 'StandardScaler', 'RobustScaler']
CLASSIFIERS = ['MLPC', 'SVC']

PREP_METHOD = METHODS[1]
CLS_METHOD = CLASSIFIERS[1]
ROOT = CLS_METHOD + '-' + PREP_METHOD

# Define the directories used in the project
DIR = {
    'raw_data': 'resource',
    'raw_data_img': os.path.join('img', 'raw_data'),
    'csv_dir': os.path.join(ROOT, 'data'),
    'conf_dir': os.path.join(ROOT, 'config'),
    'preprocessed_data_img': os.path.join(ROOT, 'img', 'preprocessed_data'),
    'results': os.path.join(ROOT, 'results'),
    'explanation': os.path.join(ROOT, 'img', 'explained_CF')
}

# Define the file names used in the project
FILE_NAME = {
    'dataset': 'K-EmoCon_15.csv',
    'prep': 'preprocessedDataset.csv',
    'params': 'modelConfiguration.json',
    'model': 'fitted_model.sav'
}

# Define the paths used in the project
PATH = {
    'dataset': os.path.join(DIR['raw_data'], FILE_NAME['dataset']),
    'preprocessed_dataset': os.path.join(DIR['csv_dir'], FILE_NAME['prep']),
    'parameters': os.path.join(DIR['conf_dir'], FILE_NAME['params']),
    'model': os.path.join(DIR['conf_dir'], FILE_NAME['model']),
    'results': os.path.join(DIR['results'], 'accuracy.txt'),
    'x_train': os.path.join(DIR['csv_dir'], 'x_train.csv'),
    'x_test': os.path.join(DIR['csv_dir'], 'x_test.csv'),
    'y_train': os.path.join(DIR['csv_dir'], 'y_train.csv'),
    'y_test': os.path.join(DIR['csv_dir'], 'y_test.csv')
}

# Generate each directory used if not already exists
for k, d in DIR.items():
    if not os.path.exists(d):
        os.makedirs(d)

# If the dataset does not exist, stop the program
if not os.path.exists(PATH['dataset']):
    print(f'ERROR: {PATH["dataset"]} does not exist.')
    sys.exit(1)
