"""
This module contains scripts for generating post-process plots.

The scripts in this module are used to visualize the results 
of a machine learning process using histograms.

"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from params.path import CLASSIFIERS, METHODS, ROOT

# Load the results
results = {}
for c in CLASSIFIERS:
    for m in METHODS:
        acc_file = os.path.join(ROOT, f'{c}-{m}', 'results', 'accuracy.txt')
        # If the file does not exist, skip
        if not os.path.exists(acc_file):
            continue
        accuracy = pd.read_csv(acc_file, header=None)
        accuracy.columns = ['accuracy']
        results[f'{c}-{m}'] = accuracy['accuracy'].mean()

# Plot the bar chart
sns.barplot(y=list(results.keys()), x=list(results.values()))
plt.title('Accuracy of different classifiers and methods')
plt.xlabel('Accuracy')
plt.ylabel('Classifier-Method')
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_score', 'accuracy_hist.png'))
