"""Compute the Jaccard measure between two model counterfactuals"""

import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cfnow import find_tabular
from lib.data_preprocessor import DataPreprocessor
from lib.classifier import Classifier
from lib.utils.path import PATH, CLASSIFIERS, PREP_METHOD
from lib.utils.attribute_specifications import ATTRIBUTES, CLASS_LABELS, DATA_LABELS

warnings.filterwarnings("ignore", category=UserWarning)


def seconds(end, start):
    """Print seconds elapsed since execution began"""
    hrs = (end - start) // 3600
    minutes = ((end - start) % 3600) // 60
    sec = (end - start) % 60
    print(f"{str(__file__).rsplit('/', maxsplit=1)[-1]} "
          f"- Execution time: "
          f"{int(hrs)} hour(s) {int(minutes)} minute(s) {sec:.3f} second(s).")


def jaccard_similarity(*sets):
    """Compute jaccard value for n sets"""
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    if len(union) == 0:
        return None
    return len(intersection) / len(union)


# Start timing
START_TIME = time.time()

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
        seconds(time.time(), START_TIME)
        sys.exit(1)
print('Models successfully loaded.')

# Sample PERCENTAGE data
PERCENTAGE = 0.1
n = int(np.round(dataset.shape[0] * PERCENTAGE))
SAMPLES = sorted(dataset.sample(n=n, random_state=42).index)
print(f"Selected indexes: {len(SAMPLES)}")

# Delete samples already computed
try:
    read_df = pd.read_csv('jaccard_indexes.csv', index_col=0)
except (FileNotFoundError, pd.errors.EmptyDataError):
    print("File not found.")
else:
    read_samples = set(read_df.index)
    SAMPLES = [data for data in SAMPLES if data not in read_samples]

# Set time limit for cf optimization
TIMEOUT = 15

# For result saving
SAVE_FREQ = 5
i = 0

# Compute jaccard indexes
jaccard_dict = {}
for sample in SAMPLES:
    print(f"Computing jaccard index for sample {sample}...")

    # Generating sets
    sampled_data = pd.Series(X.iloc[sample], name='Factual')
    feat_types = {idx: 'num' for idx in data_columns}
    sample_sets = []
    for c in clfs:
        # Generate cf object
        print(f'Generating cf obj (model: {c.class_method})...')
        cf_obj = find_tabular(
            factual=sampled_data,
            feat_types=feat_types,
            model_predict_proba=c.model.predict_proba,
            limit_seconds=TIMEOUT
        )

        # Computing changed features
        if len(cf_obj.cfs) != 0:
            cf_data = pd.Series(cf_obj.cfs[0], index=data_columns, name='CounterFact')
            diff = pd.Series((cf_data - X.iloc[sample]), index=data_columns, name='Difference')
            feature_set = set(diff[diff != 0].index)
            print(f"Changed Feature(s): {feature_set}")
        else:
            print("No counterfactual found!")
            feature_set = set()

        # Adding set to array
        sample_sets.append(feature_set)

    # Compute jaccard index
    jaccard_index = jaccard_similarity(*sample_sets)

    # Save result
    jaccard_dict[sample] = {}
    jaccard_dict[sample]['jaccard_index'] = jaccard_index
    jaccard_dict[sample]['original_class'] = y.iloc[sample]['self_valence']
    for k, s in enumerate(sample_sets):
        jaccard_dict[sample][str(clfs[k].class_method) + '_class'] = \
            np.argmax(clfs[k].model.predict_proba([sampled_data])) + 1
        jaccard_dict[sample][str(clfs[k].class_method) + '_features'] = s

    print(f"Jaccard index for sample {sample}: {jaccard_dict[sample]['jaccard_index']}")

    # Save results every SAVE_FREQ samples
    i += 1
    if i == SAVE_FREQ:
        print(f"SAVING LASTS {SAVE_FREQ} SAMPLE(S)...")
        jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient='index')
        try:
            read_df = pd.read_csv('jaccard_indexes.csv', index_col=0, header=0)
            jaccard_df.to_csv('jaccard_indexes.csv', mode='a', header=False)
            print(f"{read_df.shape[0] + i} SAMPLES DONE "
                  f"({round((read_df.shape[0] + i) * 100 / dataset.shape[0], 2)}%) "
                  f"REMAINING: {n - read_df.shape[0] - i}")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            jaccard_df.to_csv('jaccard_indexes.csv',
                              index_label='sample', header=True)
            print(f"{i} SAMPLES DONE "
                  f"({round(i * 100 / dataset.shape[0], 2)}%) "
                  f"REMAINING: {n - i}")
        i = 0
        jaccard_dict = {}

# Save last results
print(f"SAVING LASTS {i} SAMPLE(S)...")
jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient='index')
try:
    read_df = pd.read_csv('jaccard_indexes.csv', index_col=0, header=0)
    jaccard_df.to_csv('jaccard_indexes.csv', mode='a', header=False)
except (FileNotFoundError, pd.errors.EmptyDataError):
    jaccard_df.to_csv('jaccard_indexes.csv', index_label='sample', header=True)
print(f"COMPUTED JACCARD INDEX FOR ALL {len(SAMPLES)} SAMPLES!")

# Read results
result_df = pd.read_csv('jaccard_indexes.csv', index_col=0, header=0)
print(result_df)

# Compute media and std.dev
mean = result_df['jaccard_index'].mean()
std_dev = result_df['jaccard_index'].std()
print(f"Mean: {round(mean, 3)}")
print(f"Standard dev.: {round(std_dev, 3)}")

# Histogram plot
plt.figure()
plt.hist(result_df['jaccard_index'], bins=20, color='blue', alpha=0.7)
# Mean and std.dev lines
plt.axvline(x=mean, color='r', linestyle='-', label='Mean', alpha=0.7)
plt.axvline(x=mean + std_dev, color='g', linestyle='--', label='Mean +/- std.dev', alpha=0.5)
plt.axvline(x=mean - std_dev, color='g', linestyle='--', alpha=0.5)
# Text
plt.text(mean + 0.01, 50,
         f'Mean: {round(mean, 3)}', rotation=90, verticalalignment='bottom')
plt.text(mean + std_dev + 0.01, 50,
         f'Std.dev: {round(std_dev, 3)}', rotation=90, verticalalignment='bottom')
# Labels
plt.legend(loc='best')
plt.title('Jaccard index histogram')
plt.xlabel('Jaccard index')
plt.ylabel('Frequency')
# Save
plt.savefig('jaccard_indexes_histogram.png')
plt.close()

# End timing
seconds(time.time(), START_TIME)
