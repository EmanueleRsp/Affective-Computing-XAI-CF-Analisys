"""Compute the Jaccard measure between two model counterfactuals"""

import sys
import pandas as pd
from lib.timer import Timer
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from lib.jaccard_evaluer import JaccardEvaluer
from lib.utils.path import PATH, CLASSIFIERS, PREP_METHOD
from lib.utils.attribute_specifications import ATTRIBUTES, CLASS_LABELS, DATA_LABELS

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
data_columns = [ATTRIBUTES[sample] for sample in iter(DATA_LABELS)]
X = dataset.drop(columns=class_columns, axis=1)
y = dataset[class_columns]

# Load models
clfs = []
for model_type in CLASSIFIERS:
    c = Classifier(dataset, model_type)
    clfs.append(c)
    if not c.load_config():
        print(f'ERROR: Model not found for {model_type} with {PREP_METHOD} '
              'pre-processing method, please execute "model_generation.py" before.')
        timer.end()
        sys.exit(1)
print('Models successfully loaded.')

# Compute Jaccard values
j = JaccardEvaluer(
    data={'dataset': dataset, 'X': X, 'y': y},
    path='jaccard_indexes.csv'
)
j.sample_data()
j.compute_jaccard(clfs=clfs)
j.plot_jaccard_hist(clfs=clfs)
j.plot_feature_importance(clfs=clfs)

# End timing
timer.end()
