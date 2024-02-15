"""Main file for the program."""

import sys
import time
import pandas as pd
from lib.classifier import Classifier
from lib.utils.path import PATH
from lib.data_explorer import DataExplorer
from lib.data_preprocessor import DataPreprocessor
from lib.utils.path import DIR

FORCE_MODEL_UPDATE = True
STEPS_LIMIT = 3
EXPLORE_RAW_DATA = False
EXPLORE_PREPROCESSED_DATA = False
TESTS_NUMBER = 20

# Start program
start_time = time.time()

# Read the dataset
print('Reading the dataset...')
dataset = pd.read_csv(PATH['dataset'])

# Explore the data
if EXPLORE_RAW_DATA:
    start = time.time()
    de = DataExplorer(dataset, DIR['raw_data_img'])
    de.exploration()
    end = time.time()
    print(f'Raw data exploration: {end - start :.3f} seconds.')
if STEPS_LIMIT <= 1:
    # End program
    end_time = time.time()
    print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
          f"- Execution time: {end_time - start_time:.3f} seconds.")
    sys.exit()

# Preprocess the data
start = time.time()
dp = DataPreprocessor(dataset)
dataset = dp.preprocess()
end = time.time()
print(f'Data preprocessing: {end - start :.3f} seconds.')
if EXPLORE_PREPROCESSED_DATA:
    start = time.time()
    de = DataExplorer(dataset, DIR['preprocessed_data_img'])
    de.exploration()
    end = time.time()
    print(f'Preprocessed data exploration: {end - start :.3f} seconds.')
if STEPS_LIMIT == 2:
    # End program
    end_time = time.time()
    print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
          f"- Execution time: {end_time - start_time:.3f} seconds.")
    sys.exit()

for i in range(TESTS_NUMBER):
    # Generate classifier and partition the data
    c = Classifier(dataset)
    c.segregation()

    # Look for an existing model
    if not c.load_config() or (FORCE_MODEL_UPDATE and i == 0):
        # Perform classification
        start = time.time()
        c.grid_search()
        end = time.time()
        print(f'Classification completed in {end - start :.3f} seconds.')
        # Save the model
        c.save_config()

    # Calculate the accuracy of the model
    c.calculate_accuracy()
    # Generate data results
    c.generate_results()

# End program
end_time = time.time()
print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
      f"- Execution time: {end_time - start_time:.3f} seconds.")
