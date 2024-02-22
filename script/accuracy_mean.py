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
for m in METHODS:
    results[m] = {}
    for c in CLASSIFIERS:
        acc_file = os.path.join(ROOT, f'{c}-{m}', 'results', 'accuracy.txt')
        # If the file does not exist, skip
        if not os.path.exists(acc_file):
            continue
        accuracy = pd.read_csv(acc_file, header=None)
        accuracy.columns = ['accuracy']
        results[m][c] = accuracy['accuracy'].mean()

# Plot the bar chart, grouped by method using seaborn
print(results)
df = pd.DataFrame(results).T.reset_index()
df = pd.melt(df, id_vars='index', value_vars=CLASSIFIERS)
df.columns = ['method', 'classifier', 'accuracy']
ax = sns.barplot(x='accuracy', y='method', hue='classifier', data=df)
plt.title('Accuracy of different classifiers and methods')
plt.xlabel('Accuracy')
plt.ylabel('Classifier-Method')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_score', 'accuracy_hist.png'))
