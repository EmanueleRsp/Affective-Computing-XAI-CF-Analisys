"""
This module contains the DataPreprocessor class.

The DataPreprocessor class is used to clean, reduce, and transform data 
before it is used for classification. It includes methods for 
handling missing values, encoding categorical variables, normalizing numerical variables, 
and other preprocessing tasks.

Typical usage example:

    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> preprocessor = DataPreprocessor(data)
    >>> preprocessed_data = preprocessor.preprocess('prep_method')
"""

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from params.attribute_specifications import ATTRIBUTES, DATA_LABELS


class DataPreprocessor:
    """Preprocess the data.

    This class provides methods to preprocess the data, including removing missing values,
    reducing the data, normalizing the data, and saving the preprocessed data.

    Attributes:
        data (DataFrame): The input dataset to be preprocessed.

    Methods:
        __init__(self, data): Initializes the DataPreprocessor object.
        preprocess(self): Preprocesses the data and returns the preprocessed dataset.
        reduce_data(self): Reduces the data by removing unnecessary columns.
        normalize_data(self): Normalizes the data using different scaling methods.
        save_preprocessed_data(self): Saves the preprocessed data to a CSV file.
    """

    def __init__(self, data, path='preprocessedDataset.csv'):
        """Initialize the DataPreprocessor class.

        Args:
            data (DataFrame): The input data to be preprocessed.
            path (str): File path to save preprocessed data.
        """

        self.data = data
        self.path = path
        self.prep_method = None

    def preprocess(self, prep_method):
        """Preprocess the data.
        
        Args:
            prep_method (str): method used to scale data

        Returns:
            :return dataFrame: the preprocessed dataset.
        """
        print('Preprocessing the data...')
        self.prep_method = prep_method
        # Remove missing values
        self.data.dropna(inplace=True)
        # Reduce data
        self.reduce_data()
        # Normalize data
        self.normalize_data()
        # Save preprocessed data
        self.save_preprocessed_data()
        return self.data

    def reduce_data(self):
        """Reduce the data.

        This method reduces the data by removing columns before the 'self_valence' column.
        """
        print('Reducing the data...')

        # Remove columns before 'self_valence' column
        sv_index = next((i for i, v in ATTRIBUTES.items() if v == 'self_valence'), None)
        self.data = self.data.iloc[:, sv_index:]

    def normalize_data(self):
        """Transform the data.

        This method applies scaling to the numerical columns of the data
        using the specified scaling method. It iterates over the data columns
        and applies the appropriate scaler based on the self.prep_method.
        The scaled values are then assigned back to the respective columns in the data.

        Returns:
            None
        """

        print('Transforming the data...')

        # Normalize data
        data_columns = [ATTRIBUTES[i] for i in iter(DATA_LABELS)]
        for label in data_columns:
            if self.prep_method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif self.prep_method == 'RobustScaler':
                scaler = RobustScaler()
            elif self.prep_method == 'StandardScaler':
                scaler = StandardScaler()
            else:
                return
            self.data[label] = scaler.fit_transform(self.data[label].values.reshape(-1, 1))

    def save_preprocessed_data(self):
        """Save the preprocessed data.

        This method saves the preprocessed data to a CSV file.
        The file path is specified by path.
        The data is saved without including the index column.
            
        """
        print('Saving the preprocessed data...')
        self.data.to_csv(self.path, index=False)
