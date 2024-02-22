"""
Defines directories and paths managed in the project.
================================================

This file defines the directories and paths used in the project. 
It includes definitions for various directories such as raw_data, 
raw_data_img, csv_dir, conf_dir, preprocessed_data_img, results, 
and explanation.

It also defines file names such as dataset, prep, params, and model. 

Additionally, it defines paths for dataset, preprocessed_dataset, 
parameters, model, results, x_train, x_test, y_train, and y_test.

The file also includes code to generate each directory if it does not already exist, 
and checks if the dataset exists before continuing the program.

"""

import os
import sys

# Define options for the method and classifier to use in the project
METHODS = ['No normalization', 'MinMaxScaler', 'StandardScaler', 'RobustScaler']
CLASSIFIERS = ['MLPC', 'SVC']

# Define the method and classifier used in the project
PREP_METHOD = METHODS[0]
CLS_METHOD = CLASSIFIERS[0]
ROOT = 'works'
WORK_DIR = os.path.join(ROOT, f'{CLS_METHOD}-{PREP_METHOD}')

# Define the directories used in the project
DIR = {
    'raw_data': 'resource',
    'raw_data_img': os.path.join('img', 'raw_data'),
    'csv_dir': os.path.join(WORK_DIR, 'data'),
    'conf_dir': os.path.join(WORK_DIR, 'config'),
    'preprocessed_data_img': os.path.join(WORK_DIR, 'img', 'preprocessed_data'),
    'results': os.path.join(WORK_DIR, 'results'),
    'explanation': os.path.join(WORK_DIR, 'img', 'explained_CF')
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
    'accuracy': os.path.join(DIR['results'], 'accuracy.txt'),
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
