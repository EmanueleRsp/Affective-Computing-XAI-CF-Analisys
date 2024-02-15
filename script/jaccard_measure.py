"""Compute the Jaccard measure between two model counterfactuals"""

import time
import pandas as pd
from lib.utils.path import PATH
from lib.data_preprocessor import DataPreprocessor

# Start timing
start_time = time.time()

# Load the dataset
dataset = pd.read_csv(PATH['dataset'])

# Preprocess the dataset
dp = DataPreprocessor(dataset)
dataset = dp.preprocess()

print(dataset.shape)

# End timing
end_time = time.time()
print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
      f"- Execution time: {end_time - start_time:.3f} seconds.")
