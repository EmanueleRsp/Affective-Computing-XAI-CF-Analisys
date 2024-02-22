"""
This module contains the main script for generating machine learning models.

The script in this module is used to train machine learning models on a given dataset.
It includes steps for data preprocessing, model training, hyper-parameter tuning,
model evaluation, and saving the trained model.

"""

import sys
import pandas as pd
from lib.timer import Timer
from lib.data_explorer import DataExplorer
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from params.path import PATH, PREP_METHOD, DIR, CLS_METHOD

FORCE_MODEL_UPDATE = False
STEPS_LIMIT = 3
EXPLORE_RAW_DATA = False
EXPLORE_PREPROCESSED_DATA = False
TESTS_NUMBER = 20

# Start timing
timer = Timer()
timer.start()

# Read the dataset
print('Reading the dataset...')
dataset = pd.read_csv(PATH['dataset'])

# Explore the data
if EXPLORE_RAW_DATA:
    e_timer = Timer('Raw data exploration')
    e_timer.start()
    de = DataExplorer(
        data=dataset,
        save_dir=DIR['raw_data_img']
    )
    de.exploration()
    e_timer.end()
if STEPS_LIMIT <= 1:
    # End program
    timer.end()
    sys.exit()

# Preprocess the data
p_timer = Timer('Data preprocessing')
p_timer.start()
dp = DataPreprocessor(
    data=dataset,
    path=PATH['preprocessed_dataset']
)
dataset = dp.preprocess(PREP_METHOD)
p_timer.end()
if EXPLORE_PREPROCESSED_DATA:
    ep_timer = Timer('Preprocessed data exploration')
    ep_timer.start()
    de = DataExplorer(
        data=dataset,
        save_dir=DIR['preprocessed_data_img'],
        raw_dir=DIR['raw_data_img'],
        prep_method=PREP_METHOD
    )
    de.exploration()
    ep_timer.end()
if STEPS_LIMIT == 2:
    # End program
    timer.end()
    sys.exit()

for i in range(TESTS_NUMBER):
    # Generate classifier and partition the data
    c = Classifier(dataset, CLS_METHOD)
    c.segregation()

    # Look for an existing model
    if not c.load_config(PATH['model'],
                         PATH['parameters']
                         ) or (FORCE_MODEL_UPDATE and i == 0):
        # Perform classification
        g_timer = Timer('Grid Search')
        g_timer.start()
        c.grid_search()
        g_timer.end()
        # Save the model
        c.save_config(PATH['model'], PATH['parameters'])

    # Calculate the accuracy of the model
    c.calculate_accuracy()
    # Generate data results
    c.generate_results(DIR['results'], PATH['accuracy'])

# End timing
timer.end()
