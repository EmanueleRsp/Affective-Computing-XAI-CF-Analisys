"""
This module contains scripts for generating post-process plots.

The scripts in this module are used to visualize the results 
of a hyper-params optimization process using histograms.

"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mlpc = [{
    'name': 'No normalization',
    'params': [100, 200, 300],
    'mean_test_score': [0.49728951, 0.52151325, 0.52151325],
}, {
    'name': 'StandardScaler',
    'params': [100, 200, 300],
    'mean_test_score': [0.99863462, 0.99892091, 0.99892091],
}, {
    'name': 'MinMaxScaler',
    'params': [100, 200, 300],
    'mean_test_score': [0.97350676, 0.99011189, 0.99577168],
}, {
    'name': 'RobustScaler',
    'params': [100, 200, 300],
    'mean_test_score': [0.99898696, 0.99916315, 0.999163158],
}]

svc = [{
    'name': 'No normalization',
    'params': [1, 10, 100],
    'mean_test_score': [0.45408295, 0.49255634, 0.52063513],
}, {
    'name': 'StandardScaler',
    'params': [1, 10, 100],
    'mean_test_score': [0.97843989, 0.99517704, 0.99841438],
}, {
    'name': 'MinMaxScaler',
    'params': [1, 10, 100],
    'mean_test_score': [0.91402402, 0.96273789, 0.98978154],
}, {
    'name': 'RobustScaler',
    'params': [1, 10, 100],
    'mean_test_score': [0.91961778, 0.97586325, 0.99623414],
}]

# Generate results
params = [1, 10, 100]
results = {}
for idx, p in enumerate(params):
    results[p] = {}
    for method in svc:
        results[p][method['name']] = round(method['mean_test_score'][idx], 3)

# Plot the bar chart, grouped by params using seaborn
df = pd.DataFrame(results)
df = df.T
print(df)
df = df.reset_index()
df = df.rename(columns={'index': 'params'})
df = pd.melt(df, id_vars='params',
             value_vars=['No normalization',
                         'StandardScaler',
                         'MinMaxScaler',
                         'RobustScaler']
             )
df = df.rename(columns={'variable': 'Normalization', 'value': 'Mean test score'})
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='params', y='Mean test score', hue='Normalization', data=df)
plt.title('MLPClassifier mean test score')
plt.xlabel('C parameter')
plt.ylabel('Mean test score')
plt.legend(loc='lower right')
plt.savefig(os.path.join('img', 'model_score', 'grid_search_SVC.png'))
plt.close()
