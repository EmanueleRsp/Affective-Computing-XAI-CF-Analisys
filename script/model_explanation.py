"""Use cfnow library to plot some counterfactual explanations on the model trained"""

import time
import sys
import pandas as pd
from lib.explainer import Explainer
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from lib.utils.path import PATH, PREP_METHOD, CLS_METHOD
from lib.utils.attribute_specifications import ATTRIBUTES, CLASS_LABELS

# Start timing
start_time = time.time()

# Load the dataset
dataset = pd.read_csv(PATH['dataset'])

# Preprocess the dataset
dp = DataPreprocessor(dataset)
dataset = dp.preprocess()

# Load the model
c = Classifier(dataset)
if not c.load_config():
    print(f'ERROR: Model not found for {CLS_METHOD} with {PREP_METHOD} '
          'pre-processing method, please execute "model_generation.py" before.')
    end_time = time.time()
    print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
          f"- Execution time: {end_time - start_time:.3f} seconds.")
    sys.exit(1)
print('Model successfully loaded.')

# Divide dataset in features and targets
class_columns = [ATTRIBUTES[sample] for sample in iter(CLASS_LABELS)]
X = dataset.drop(columns=class_columns, axis=1)
y = dataset[class_columns]

# Sample some data for each class
SAMPLES = [1, 9, 50, 70, 99, 127, 1000, 1100, 1300, 1910, 1916,
           1919, 1925, 7711, 19840, 27309, 29971, 33962, 39082, 45981]
# for value in dataset['self_valence'].unique():
#     SAMPLES.extend(dataset[dataset['self_valence'] == value].sample(n=10, random_state=42).index)
# Set time limit
TIMEOUT = 30
# Set plot mod
# MOD = ['greedy', 'countershapley']

# Explain data
e = Explainer(SAMPLES, {'X': X, 'y': y}, c.model, TIMEOUT)
e.plot_counterfactuals()

# End timing
end_time = time.time()
print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
      f"- Execution time: {end_time - start_time:.3f} seconds.")
