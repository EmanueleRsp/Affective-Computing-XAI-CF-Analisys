"""
This module contains the JaccardEvaluator class.

The JaccardEvaluator class is used to compute the Jaccard similarity coefficient
between the predictions of different classifiers. This coefficient measures 
the similarity between two sets and is defined as the size of the intersection 
divided by the size of the union of the sets.

Typical usage example:

    >>> import pandas
    >>> data = pandas.read_csv('data.csv')
    >>> X = pandas.read_csv('X.csv')
    >>> y = pandas.read_csv('y.csv')
    >>> evaluator = JaccardEvaluer(
    >>>     data={'dataset': data, 'X': X, 'y': y},
    >>>     path='path'
    >>> )
    >>> evaluator.compute_jaccard(clfs)
"""
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cfnow import find_tabular
from params.attribute_specifications import ATTRIBUTES, DATA_LABELS

warnings.filterwarnings("ignore", category=UserWarning)


def jaccard_similarity(*sets):
    """
    Compute the Jaccard similarity coefficient for a given set of sets.

    Parameters:
    sets (tuple): A tuple of sets for which the Jaccard similarity coefficient needs to be computed.

    Returns:
    float: The Jaccard similarity coefficient, which is a value between 0 and 1. 
        If the union of all sets is empty, returns None.
    """
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    if len(union) == 0:
        return None
    return len(intersection) / len(union)


class JaccardEvaluer:
    """
    The JaccardEvaluer class is used to measure the Jaccard values between classifiers.

    Attributes:
        data (dict): Contains dataset, X, y.
        path (str): The path to save the evaluation results (csv file).
        samples (list): List of selected samples.
        percentage (float): The percentage of data to sample.
        timeout (int): Timeout value for computation.
        save_freq (int): Frequency of saving results.

    Methods:
        __init__(self, data, path='jaccard_indexes.csv'): Initializes the JaccardEvaluer class.
        sample_data(self, percentage=0.1): Samples the data.
        compute_jaccard(self, clfs, timeout=15, save_freq=25):
            Computes Jaccard values between classifiers.
    """

    # Data columns labels for the dataset
    data_columns = [ATTRIBUTES[sample] for sample in iter(DATA_LABELS)]

    def __init__(self, data, path='jaccard_indexes.csv'):
        """
        Initialize the JaccardEvaluer class.

        Parameters:
            data (dict): Contains dataset, X, y.
            path (str): The path to save the evaluation results (csv file).

        Returns:
            None
        """

        self.data = data
        self.path = path
        self.samples = None
        self.percentage = None
        self.timeout = None
        self.save_freq = None

    def sample_data(self, percentage=0.1):
        """Sample the data

        This method samples the data based on the specified percentage. 
        It selects a subset of the dataset randomly and sets it 
        as the samples attribute of the JaccardEvaluer object.

        Parameters:
            percentage (float): The percentage of data to sample. Defaults to 0.1.

        Returns:
            None
        """

        # Initialize percentage
        self.percentage = percentage

        # Choose samples
        sample_size = int(np.round(self.data['dataset'].shape[0] * self.percentage))
        self.samples = self.data['dataset'].sample(n=sample_size, random_state=42).index
        print(f"Selected indexes: {len(self.samples)}")

        # Delete samples already computed
        try:
            read_df = pd.read_csv(self.path, index_col=0)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("File not found.")
        else:
            read_samples = set(read_df.index)
            self.samples = [data for data in self.samples if data not in read_samples]

    def compute_jaccard(self, clfs, timeout=15, save_freq=25):
        """Compute Jaccard values between classifiers

        This method computes the Jaccard values between classifiers for each sample in the dataset.
        It generates counterfactual objects using the find_tabular function and calculates the
        Jaccard index based on the changed features. The results are saved periodically and
        the final results are printed once all samples are processed.

        Parameters:
            clfs (list): List of classifiers.
            timeout (int): Timeout value for computation. Defaults to 15.
            save_freq (int): Frequency of saving results. Defaults to 25.

        Returns:
            None
        """

        self.timeout = timeout
        self.save_freq = save_freq

        jaccard_dict = {}
        i = 0
        for sample in self.samples:
            print(f"Computing jaccard index for sample {sample}...")

            # Generating sets
            sampled_data = pd.Series(self.data['X'].iloc[sample], name='Factual')
            feat_types = {idx: 'num' for idx in self.data_columns}
            sample_sets = []
            for classifier in clfs:
                # Generate cf object
                print(f'Generating cf obj (model: {classifier.class_method})...')
                cf_obj = find_tabular(
                    factual=sampled_data,
                    feat_types=feat_types,
                    model_predict_proba=classifier.model.predict_proba,
                    limit_seconds=self.timeout
                )

                # Computing changed features
                if len(cf_obj.cfs) != 0:
                    cf_data = pd.Series(cf_obj.cfs[0], index=self.data_columns, name='CounterFact')
                    diff = pd.Series((cf_data - self.data['X'].iloc[sample]),
                                     index=self.data_columns, name='Difference')
                    feature_set = set(diff[diff != 0].index)
                    print(f"Changed Feature(sets): {feature_set}")
                else:
                    print("No counterfactual found!")
                    feature_set = set()

                # Adding set to array
                sample_sets.append(feature_set)

            # Create dict with results info
            jaccard_dict[sample] = self._jaccard_dict_row(
                sample=sample,
                sample_sets=sample_sets,
                clfs=clfs
            )

            # Save results every self.save_freq samples
            i += 1
            if i == self.save_freq:
                self._save_results(
                    jaccard_dict=jaccard_dict,
                    sample_num=i
                )
                i = 0
                jaccard_dict = {}

        # Save last results
        if i != 0:
            self._save_results(
                jaccard_dict=jaccard_dict,
                sample_num=i
            )
        print(f"COMPUTED JACCARD INDEX FOR ALL {len(self.samples)} SAMPLES!")

    def plot_jaccard_hist(self, clfs, path='jaccard_indexes_histogram.png'):
        """Plot Jaccard results

        This method plots the Jaccard results by creating
        a bar plot of the feature importance.
        It reads the results from the specified path and computes
        the common features for each classifier.
        The resulting bar plot shows the frequency of each feature
        for each classifier.

        Parameters:
            clfs (list): List of classifiers.
            path (str): The path to save the plot.

        Returns:
            None
        """

        print("Plotting jaccard index values distribution...")

        # Read results
        result_df = pd.read_csv(self.path, index_col=0, header=0)

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
        plt.axvline(x=mean + std_dev, color='g', linestyle='--',
                    label='Mean +/- std.dev', alpha=0.5)
        plt.axvline(x=mean - std_dev, color='g', linestyle='--', alpha=0.5)
        # Text
        plt.text(mean + 0.01, 50,
                 f'Mean: {round(mean, 3)}', rotation=90, verticalalignment='bottom')
        plt.text(mean + std_dev + 0.01, 50,
                 f'Std.dev: {round(std_dev, 3)}', rotation=90, verticalalignment='bottom')
        # Labels
        plt.legend(loc='best')
        plt.title(f"Distribution of Jaccard index between "
                  f"{', '.join([classifier.class_method for classifier in clfs])}")
        plt.xlabel('Jaccard index')
        plt.ylabel('Frequency')
        # Save
        plt.savefig(path)
        plt.close()

    def plot_feature_importance(self, clfs, path='features_importance.png'):
        """Plot feature importance

        This method plots the feature importance by creating a bar plot.
        It reads the results from the specified path and computes the
        common features for each classifier.
        The resulting bar plot shows the frequency of each feature for each classifier.

        Parameters:
            clfs (list): List of classifiers.
            path (str): The path to save the plot.

        Returns:
            None
        """

        print("Plotting features importance...")

        # Read results
        result_df = pd.read_csv(self.path, index_col=0, header=0)

        common_features = {}
        for classifier in clfs:
            common_features[classifier.class_method] = \
                {feat: sum(feat in row for row in result_df[classifier.class_method + '_features'])
                 for feat in self.data_columns}

        # Create dataframe
        data = []
        for model, feat in common_features.items():
            for key, value in feat.items():
                data.append([model, key, value])
        data = pd.DataFrame(data, columns=['Model', 'Feature', 'Frequency'])

        # Plot
        sns.barplot(x='Frequency', y='Feature', hue='Model', data=data)
        plt.title('Features importance')
        plt.xlabel('Frequency')
        plt.ylabel('Features')
        plt.tight_layout()
        # Save
        plt.savefig(path)

    def _save_results(self, jaccard_dict, sample_num):
        """Save jaccard indexes computed to file

        This method saves the computed Jaccard indexes to a file.
        It takes a dictionary of Jaccard indexes and the number of samples as input.
        The method appends the Jaccard indexes to an existing file or creates
        a new file if it doesn't exist.
        It also prints the progress of saving the samples.

        Parameters:
            jaccard_dict (dict): Dictionary of Jaccard indexes.
            sample_num (int): Number of samples.

        Returns:
            None
        """

        print(f"SAVING LASTS {sample_num} SAMPLE(S)...")

        # Total number of samples to compute
        sample_size = int(np.round(self.data['dataset'].shape[0] * self.percentage))

        # Create dataframe
        jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient='index')
        try:
            # If file already exists and not empty, append results
            read_df = pd.read_csv(self.path, index_col=0, header=0)
            jaccard_df.to_csv(self.path, mode='a', header=False)
            perc = round((read_df.shape[0] + sample_num) * 100 / self.data['dataset'].shape[0], 2)
            print(f"{read_df.shape[0] + sample_num} self.samples DONE "
                  f"({perc}%) "
                  f"REMAINING: {sample_size - read_df.shape[0] - sample_num}")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Create file if not exists or empty
            jaccard_df.to_csv(self.path,
                              index_label='sample', header=True)
            print(f"{sample_num} self.samples SAVED "
                  f"({round(sample_num * 100 / self.data['dataset'].shape[0], 2)}%) "
                  f"REMAINING: {sample_size - sample_num}")

    def _jaccard_dict_row(self, sample, sample_sets, clfs):
        """Create dict for Jaccard index computed on a sample

        This method takes a sample, sample sets, and classifiers
        as input and computes the Jaccard index for the sample.
        It creates a dictionary containing the Jaccard index,
        the original class of the sample, and the predicted class
        and set features for each model.

        Parameters:
            sample (int): The index of the sample.
            sample_sets (list): List of sets for the sample.
            clfs (list): List of classifiers.

        Returns:
            dict: Dictionary containing the Jaccard index,
            original class, predicted class, and set features for each model.
        """

        # Compute jaccard index
        jaccard_index = jaccard_similarity(*sample_sets)
        print(f"Jaccard index for sample {sample}: {jaccard_index}")

        # Sampled data
        sampled_data = pd.Series(self.data['X'].iloc[sample], name='Factual')

        # Jaccard index and original sample class
        row = {
            'jaccard_index': jaccard_index,
            'original_class': self.data['y'].iloc[sample]['self_valence']
        }

        # Predicted class and set features for each model
        for k, sets in enumerate(sample_sets):
            row[str(clfs[k].class_method) + '_class'] = \
                np.argmax(clfs[k].model.predict_proba([sampled_data])) + 1
            row[str(clfs[k].class_method) + '_features'] = sets

        return row
