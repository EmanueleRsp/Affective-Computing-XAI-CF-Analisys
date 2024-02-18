"""
This module contains the Classifier class.

The Classifier class is used to classify data based on certain criteria or algorithms. 
It includes methods for segregating the data into training and testing sets, 
performing grid search to find the best parameters, calculating the accuracy of the model,
saving and loading the model configuration, and generating results.

Typical usage example:

    >>> clf = Classifier(data, model)
    >>> clf.segregation()
    >>> clf.grid_search()
    >>> clf.calculate_accuracy()
    >>> clf.save_config()
    >>> clf.load_config()
    >>> clf.generate_results()

"""
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
    """
    Classify the data.

    This class provides methods for classifying data using different models. It supports
    segregation of data into training and testing sets, performing grid search to find the
    best parameters, calculating the accuracy of the model, saving and loading the model
    configuration, and generating results including accuracy, best parameters, grid search
    results, and confusion matrix.

    Attributes:
        `data` (pd.DataFrame): The input data for classification.
        `partition` (dict): A dictionary containing the segregated training and testing sets.
        `class_method` (str): The classification model to be used.
        `model` (object): The classification model object.
        `grid` (dict): A dictionary containing the grid search results and best parameters.
        `score` (float): The accuracy score of the model.
        `y_pred` (array-like): The predicted labels for the testing set.

    Methods:
        `segregation()`: Segregate the data into training and testing sets.
        `_get_parameters()`: Get the parameters for the model.
        `grid_search(cval=5)`: Perform grid search to find the best parameters.
        `calculate_accuracy()`: Calculate the accuracy of the model.
        `save_config()`: Save the model configuration.
        `load_config()`: Load the model configuration.
        `generate_results()`: Generate results including accuracy, best parameters, grid search
            results, and confusion matrix.
    """

    def __init__(self, data, model):
        """Initialize the classifier.

        Args:
            data (pd.DataFrame): The input data for classification.
            model (str): The classification model to be used.
        """

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
        """Segregate the data into training and testing sets.

            This method splits the data into training and testing sets 
            using the train_test_split function from the scikit-learn library.
            It assigns the training and testing data to the 'x_train', 'x_test',
            'y_train', and 'y_test' attributes of the 'partition' dictionary.

            Returns:
                None
            """

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
        """Get the parameters for the model.

            Returns:
                dict: A dictionary containing the parameters for the model.
                      The specific parameters depend on the selected classifier.
                      For the Multi-layer Perceptron classifier, the dictionary
                      contains the 'max_iter' parameter with possible values of
                      [100, 200, 300]. For the Support Vector Classifier, the
                      dictionary contains the 'C' parameter with possible values
                      of [1, 10, 100] and the 'probability' parameter set to True.
            """

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
        """
        Perform grid search to find the best parameters.

        Parameters:
        - cval (int): Number of cross-validation folds (default: 5)
        """

        print('Performing grid search to find the best parameters and model...')
        grid = GridSearchCV(self.model, self._get_parameters(), cv=cval)
        grid.fit(self.partition['x_train'], ravel(self.partition['y_train']))
        self.model = grid.best_estimator_
        self.grid['results'] = grid.cv_results_
        self.grid['best_params'] = grid.best_params_
        print(f'Best parameters found: {self.grid["best_params"]}')

    def calculate_accuracy(self):
        """Calculate the accuracy of the model.

            Returns:
                float: The accuracy score of the model.
            """

        print('Calculating the accuracy of the model...')
        self.y_pred = self.model.predict(self.partition['x_test'])
        self.score = accuracy_score(self.partition['y_test'], self.y_pred)

    def save_config(self):
        """Save the model and its parameters.

            This method saves the trained model and its best parameters to disk.
            The model is saved using joblib.dump() function, and the parameters
            are saved in a JSON file.

            Returns:
                None
            """

        print('Saving settings...')
        joblib.dump(self.model, PATH['model'])
        with open(PATH['parameters'], 'w', encoding='utf-8') as file:
            json.dump(self.grid['best_params'], file, indent=4)

    def load_config(self):
        """
        Load the model in 'self.model',
        load the parameters which is based on in 'self.grid["best_parameters"]'.

        Returns:
            bool: True if the model exists, else False.
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
        """Generate results.

            This method generates and prints the accuracy, best parameters,
            and grid search results.
            It also saves the accuracy score to a file and generates
            a confusion matrix for each class.

            """
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
