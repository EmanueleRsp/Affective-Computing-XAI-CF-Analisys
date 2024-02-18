"""
This module contains scripts for generating post-process plots.

The scripts in this module are used to visualize the results of a machine learning process.
This includes generating plots such as bar plots, scatter plots, and others to understand
the scores, performance, and other aspects of the models used.

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

mlpc = [{
    'name': 'StandardScaler',
    'text': {'x': -10, 'y': -0.0013},
    'color': 'green',
    'values': {
        'params': [{'max_iter': 100}, {'max_iter': 200}, {'max_iter': 300}],
        'mean_test_score': [0.99863462, 0.99892091, 0.99892091],
        'rank_test_score': [3, 1, 1]
    }
}, {
    'name': 'MinMaxScaler',
    'text': {'x': -10, 'y': +0.0005},
    'color': 'yellow',
    'values': {
        'params': [{'max_iter': 100}, {'max_iter': 200}, {'max_iter': 300}],
        'mean_test_score': [0.97350676, 0.99011189, 0.99577168],
        'rank_test_score': [3, 2, 1]
    }
}, {
    'name': 'RobustScaler',
    'text': {'x': -10, 'y': 0.0005},
    'color': 'red',
    'values': {
        'params': [{'max_iter': 100}, {'max_iter': 200}, {'max_iter': 300}],
        'mean_test_score': [0.99898696, 0.99916315, 0.999163158],
        'rank_test_score': [3, 1, 1]
    }
}]

svc = [{
    'name': 'MinMaxScaler',
    'text': {'x': 0, 'y': 0},
    'color': 'red',
    'values': {
        'params': [{'C': 1}, {'C': 10}, {'C': 100}],
        'mean_test_score': [0.91402402, 0.96273789, 0.98978154],
        'rank_test_score': [3, 2, 1]
    }
}]

for result in mlpc:
    r_df = pd.DataFrame(result['values'])
    r_df['params'] = r_df['params'].apply(lambda x: x['max_iter'])

    # Using a line plot
    plt.plot(
        r_df['params'], r_df['mean_test_score'],
        marker='o', label=result['name'], color=result['color'], alpha=0.5
    )

    # Add labels to the points
    for i in range(len(r_df['params'])):
        plt.text(
            r_df['params'].iloc[i] + result['text']['x'],
            r_df['mean_test_score'].iloc[i] + result['text']['y'],
            str(round(r_df['mean_test_score'].iloc[i], 4))
        )

plt.xlabel('max_iter')
plt.ylabel('mean_test_score')
plt.title('Mean test score for different params and scaler (MLP)')
plt.legend()
plt.savefig(os.path.join('img', 'model_score', 'mlp_mean_test_score_zoom.png'))
plt.close()
