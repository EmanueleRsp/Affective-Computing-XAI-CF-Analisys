"""Compute the Jaccard measure between two model counterfactuals"""

import sys
import time
import pandas as pd
import numpy as np
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from lib.utils.path import PATH, CLASSIFIERS, PREP_METHOD


def seconds(end, start):
    """Print seconds elapsed since execution began"""
    print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
          f"- Execution time: {end - start:.3f} seconds.")


def jaccard_similarity_multi(*sets):
    """Compute jaccard value for n sets"""
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    return len(intersection) / len(union)


# Start timing
START_TIME = time.time()

# Load the dataset
dataset = pd.read_csv(PATH['dataset'])

# Preprocess the dataset
dp = DataPreprocessor(dataset)
dataset = dp.preprocess()

# Load models
clfs = []
for model_type in CLASSIFIERS:
    c = Classifier(dataset, model_type)
    clfs.append(c)
    if not c.load_config():
        print(f'ERROR: Model not found for {model_type} with {PREP_METHOD} '
              'pre-processing method, please execute "model_generation.py" before.')
        seconds(time.time(), START_TIME)
        sys.exit(1)
print('Models successfully loaded.')

# Sample PERCENTAGE data
PERCENTAGE = 0.0001
n = int(np.round(dataset.shape[0] * PERCENTAGE))
SAMPLES = sorted(dataset.sample(n=n, random_state=42).index)
print(f"Selected indexes: {SAMPLES}")

# Compute
jaccard_dict = {}
for sample in SAMPLES:
    print(f"Computing jaccard index for sample {sample}...")

# Save results
jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient='index', columns=['jaccard_index'])
print(jaccard_df)
jaccard_df.to_csv('results/jaccard_indexes.csv', index_label='Sample', columns=['jaccard_index'])

# End timing
seconds(time.time(), START_TIME)
