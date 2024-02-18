"""
This module uses the cfnow library to explain models.

The cfnow library is a powerful tool for interpreting machine learning models.
It provides methods for generating counterfactual explanations, 
which can help understand how a model makes its predictions.

Typical usage example:

    >>> explainer = cfnow.Explainer(model)
    >>> explanation = explainer.explain(data)
"""
import os
import warnings
import pandas as pd
import numpy as np
from cfnow import find_tabular
from lib.utils.path import DIR, CLS_METHOD
warnings.simplefilter(action='ignore', category=FutureWarning)


class Explainer:
    """Explain the model.

    This class provides methods to explain a machine learning model
    by computing counterfactuals and Shapley values.
    It takes a set of samples, data, model, and an optional timeout parameter
    as input during initialization.

    Attributes:
        samples (list): The indices of the samples to be explained.
        data (pandas.DataFrame): The input features.
            It contains the input features (X) and the target classes (Y).
        classes (pandas.DataFrame): The target classes.
        model (object): The machine learning model.
        timeout (float): The maximum time allowed for computing counterfactuals.

    Methods:
        plot_counterfactuals(mod=None): Compute and plot counterfactuals for each sample.
        compute_shapley_values(): Compute Shapley values for each sample.

    Private Methods:
        _generate_cf_obj(sample): Generate a counterfactual object for a given sample.
    """

    # Available graph modes
    MOD = ['greedy', 'countershapley', 'constellation']

    def __init__(self, samples, data, model, timeout=None):
        """Initialize the explainer.

        Args:
            samples (list): The indices of the samples to be explained.
            data (dict): The input features and target classes.
            model (object): The machine learning model.
            timeout (float, optional): The maximum time allowed for computing counterfactuals.
        """

        self.samples = samples
        self.data = data['X']
        self.classes = data['Y']
        self.model = model
        self.timeout = timeout

    def plot_counterfactuals(self, mod=None):
        """Compute counterfactual for each sample

        This method computes the counterfactual for each sample in the Explainer object.
        It generates counterfactual objects using the _generate_cf_obj method 
        and generates plots for each counterfactual.

        Args:
            mod (list, optional): The graph modes to be generated. 
                Defaults to None, which generates all available modes.

        Returns:
            None
        """

        # Default: all
        if mod is None:
            mod = self.MOD
        # Create dirs if not exist
        if not os.path.exists(os.path.join(DIR['explanation'], 'greedy')):
            os.makedirs(os.path.join(DIR['explanation'], 'greedy'))
        if not os.path.exists(os.path.join(DIR['explanation'], 'counterShapley')):
            os.makedirs(os.path.join(DIR['explanation'], 'counterShapley'))
        if not os.path.exists(os.path.join(DIR['explanation'], 'constellation')):
            os.makedirs(os.path.join(DIR['explanation'], 'constellation'))

        # Compute counterfactuals for each sample
        for sample in self.samples:

            # Generate cf_obj
            cf_obj, _, _ = self._generate_cf_obj(sample)
            if cf_obj is None:
                continue

            # Generating plots
            counter_plot = cf_obj.generate_counterplots(0)

            if self.MOD[0] in mod:
                print('Generating greedy counterplot...')
                counter_plot.greedy(
                    os.path.join(DIR['explanation'], 'greedy',
                                 f'{sample}_greedy_counterplot.png')
                )

            if self.MOD[1] in mod:
                print('Generating counterShapley counterplot...')
                counter_plot.countershapley(
                    os.path.join(DIR['explanation'], 'counterShapley',
                                 f'{sample}_counterShapley_counterplot.png')
                )

            if self.MOD[2] in mod:
                print('Generating constellation counterplot...')
                counter_plot.constellation(
                    os.path.join(DIR['explanation'], 'constellation',
                                 f'{sample}_constellation_counterplot.png')
                )

    def compute_shapley_values(self):
        """Compute Shapley values

        This method computes the Shapley values for each sample.
        Shapley values are a method for assigning importance scores
        to features in a model. They quantify the contribution of each feature
        towards the prediction made by the model.
        The Shapley values are computed using the counterfactual objects
        generated for each sample.

        Returns:
            None
        """

        # Choose sample already analyzed
        new_samples = []
        try:
            loaded_result = pd.read_csv(
                os.path.join(DIR['explanation'], 'shapley_values.csv'),
                index_col=0,
                header=0
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print('No data already studied.')
        else:
            idx = set(loaded_result.index)
            new_samples = [sample_data for sample_data in self.samples if sample_data not in idx]

        # Compute counterfactuals for each sample
        for sample in new_samples:

            # Generate cf_obj
            cf_obj, factual_class, counterfactual_class = self._generate_cf_obj(sample)
            if cf_obj is None:
                continue

            # Generating plots
            counter_plot = cf_obj.generate_counterplots(0)
            print("Generating countershapley values...")
            cs_values = counter_plot.countershapley_values()

            # Create a dictionary with 'sample', 'factual_class', and 'counterfactual_class'
            data = {
                'sample': sample,
                'factual_class': factual_class,
                'counterfactual_class': counterfactual_class
            }

            # Add each column from self.data.columns to the dictionary
            for column in self.data.columns:
                if column in cs_values['feature_names']:
                    index = cs_values['feature_names'].index(column)
                    data[column] = cs_values['feature_values'][index]
                else:
                    data[column] = 0

            # Append to file
            result_df = pd.DataFrame(data, index=[data['sample']])
            result_df.to_csv(
                os.path.join(DIR['explanation'], 'shapley_values.csv'),
                mode='a',
                header=False,
                index=False
            )

    def _generate_cf_obj(self, sample):
        """Generate counterfactual object

        This method generates a counterfactual object for a given sample. 
        It takes the sample index as input and returns the counterfactual object,
        the factual class, and the counterfactual class.

        Parameters:
            sample (int): The index of the sample.

        Returns:
            cf_obj (object): The counterfactual object.
            factual_class (int): The factual class of the sample.
            counterfactual_class (int): The counterfactual class of the sample.
        """

        sample_data = pd.Series(self.data.iloc[sample], name='Factual')

        # Skipping misclassified data
        # In general, model prediction and probability prediction function could return
        # different classifications. We use probability prediction because cfnow use it.
        original_label = self.classes.iloc[sample]['self_valence']
        prob_pred_label = np.argmax(self.model.predict_proba([sample_data])[0]) + 1
        cls_pred_label = self.model.predict([sample_data])[0]

        if prob_pred_label != original_label:
            print(f"Sample {sample} is misclassified by probability function "
                  f"(label: {original_label}, "
                  f"probability prediction: {prob_pred_label}, "
                  f"{CLS_METHOD} prediction: {cls_pred_label}"
                  "), move on to the next one!")
            return None, None, None

        print(f"Sample {sample}: "
              f"label={original_label}, "
              f"prob={prob_pred_label}, "
              f"class={cls_pred_label}")

        # Generate the minimum change
        print(f'Generating counterfactual object for sample {sample}...')
        cf_obj = find_tabular(
            factual=sample_data,
            feat_types={c: 'num' for c in self.data.columns},
            model_predict_proba=self.model.predict_proba,
            limit_seconds=self.timeout
        )

        # Display the new class
        cf_data = pd.Series(cf_obj.cfs[0], index=self.data.columns, name='CounterFact')
        diff = pd.Series((cf_data - sample_data), index=self.data.columns, name='Difference')
        result_df = pd.concat([sample_data, cf_data, diff], axis=1)
        factual_class = prob_pred_label
        prob_cf_class = np.argmax(self.model.predict_proba([cf_obj.cfs[0]])[0]) + 1
        cls_cf_class = self.model.predict([cf_obj.cfs[0]])[0]
        print(f"{result_df}")
        print(f"Factual class: {factual_class}")
        print(f"Counterfactual class (by probability function): {prob_cf_class}")
        print(f"Counterfactual class (by {CLS_METHOD}): {cls_cf_class}")

        return cf_obj, factual_class, prob_cf_class
