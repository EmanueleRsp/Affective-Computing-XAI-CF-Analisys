"""Contains the Classifier class, which is used to classify the data."""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import ravel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from lib.utils.path import PATH, DIR, CLASSIFIERS
from lib.utils.attribute_specifications import ATTRIBUTES, DATA_LABELS, CLASS_LABELS


class Classifier:
    """Classify the data."""

    def __init__(self, data, model):
        """Initialize the classifier."""

        self.data = data
        self.partition = {
            'x_train': pd.DataFrame(),
            'x_test': pd.DataFrame(),
            'y_train': pd.DataFrame(),
            'y_test': pd.DataFrame()
        }

        self.class_method = model
        if self.class_method == CLASSIFIERS[0]:
            self.model = MLPClassifier(random_state=0)
        else:
            self.model = SVC(random_state=0, probability=True)

        self.grid = {
            'results': None,
            'best_params': None,
        }

        self.score = None
        self.y_pred = None

    def segregation(self):
        """Segregate the data into training and testing sets."""

        print('Segregating the data into training and testing sets...')
        class_columns = [ATTRIBUTES[i] for i in iter(CLASS_LABELS)]
        data_columns = [ATTRIBUTES[i] for i in iter(DATA_LABELS)]
        [self.partition['x_train'], self.partition['x_test'],
         self.partition['y_train'], self.partition['y_test']] = \
            train_test_split(self.data[data_columns], self.data[class_columns])

        # Print shapes
        print(f'x_train shape: {self.partition["x_train"].shape}')
        print(f'x_test shape: {self.partition["x_test"].shape}')
        print(f'y_train shape: {self.partition["y_train"].shape}')
        print(f'y_test shape: {self.partition["y_test"].shape}')

    def _get_parameters(self):
        """Get the parameters for the model."""

        # Multi-layer Perceptron
        if self.class_method == CLASSIFIERS[0]:
            return {
                'max_iter': [100, 200, 300]
            }

        # Support Vector Classifier
        return {
            'C': [1, 10, 100],
            'probability': [True]
        }

    def grid_search(self, cval=5):
        """Perform grid search to find the best parameters."""
        print('Performing grid search to find the best parameters and model...')
        grid = GridSearchCV(self.model, self._get_parameters(), cv=cval)
        grid.fit(self.partition['x_train'], ravel(self.partition['y_train']))
        self.model = grid.best_estimator_
        self.grid['results'] = grid.cv_results_
        self.grid['best_params'] = grid.best_params_
        print(f'Best parameters found: {self.grid["best_params"]}')

    def calculate_accuracy(self):
        """Calculate the accuracy of the model."""

        print('Calculating the accuracy of the model...')
        self.y_pred = self.model.predict(self.partition['x_test'])
        self.score = accuracy_score(self.partition['y_test'], self.y_pred)

    def save_config(self):
        """Save the model."""

        print('Saving settings...')
        joblib.dump(self.model, PATH['model'])
        with open(PATH['parameters'], 'w', encoding='utf-8') as file:
            json.dump(self.grid['best_params'], file, indent=4)

    def load_config(self):
        """
        Load the model in 'self.model',
        load the parameters which is based on in 'self.grid["best_parameters"]'.

        Returns:
            :return bool: True if the model exists, else False.
        """

        if not os.path.exists(PATH['model']):
            print('No model found.')
            return False

        print('Loading settings...')
        self.model = joblib.load(PATH['model'])
        with open(PATH['parameters'], 'r', encoding='utf-8') as file:
            self.grid['best_params'] = json.load(file)

        return True

    def generate_results(self):
        """Generate results."""
        print('Generating results...')

        # Results
        print(f'Accuracy: {self.score:.5f}')
        print(f'Best parameters: {self.grid["best_params"]}')
        print(f'Grid search results: {self.grid["results"]}')
        with open(PATH['results'], 'a', encoding='utf-8') as file:
            file.write(f'{str(self.score)}\n')

        # Confusion matrix
        print('Generating confusion matrix...')
        class_columns = [ATTRIBUTES[i] for i in iter(CLASS_LABELS)]
        for i, label in enumerate(class_columns):
            cm_plot = confusion_matrix(self.partition['y_test'], self.y_pred)
            plt.figure(figsize=(10, 7))
            axs = sns.heatmap(cm_plot, annot=True, fmt='d')
            plt.title(f'Confusion matrix for class "{label}"')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            axs.set_xticklabels([str(int(tick.get_text()) + 1) for tick in axs.get_xticklabels()])
            axs.set_yticklabels([str(int(tick.get_text()) + 1) for tick in axs.get_yticklabels()])
            plt.savefig(os.path.join(DIR['results'], f'{label}_confusion_matrix.png'))
