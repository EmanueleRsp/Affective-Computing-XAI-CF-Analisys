"""
This module uses the cfnow library to generate and plot 
counterfactual explanations for a trained model.

The cfnow library is a powerful tool for interpreting machine learning models.
It provides methods for generating counterfactual explanations, which can help
understand how a model makes its predictions. This module uses these capabilities
to generate and plot counterfactual explanations for a trained model.

"""
import sys
import pandas as pd
from lib.timer import Timer
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from lib.explainer import Explainer
from lib.utils.path import PATH, PREP_METHOD, CLS_METHOD
from lib.utils.attribute_specifications import ATTRIBUTES, CLASS_LABELS

# Start timing
timer = Timer()
timer.start()

# Load the dataset
dataset = pd.read_csv(PATH['dataset'])

# Preprocess the dataset
dp = DataPreprocessor(dataset)
dataset = dp.preprocess()

# Divide dataset in features and targets
class_columns = [ATTRIBUTES[sample] for sample in iter(CLASS_LABELS)]
X = dataset.drop(columns=class_columns, axis=1)
y = dataset[class_columns]

# Load the model
c = Classifier(dataset, CLS_METHOD)
if not c.load_config():
    print(f'ERROR: Model not found for {CLS_METHOD} with {PREP_METHOD} '
          'pre-processing method, please execute "model_generation.py" before.')
    timer.end()
    sys.exit(1)
print('Model successfully loaded.')

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
timer.end()
