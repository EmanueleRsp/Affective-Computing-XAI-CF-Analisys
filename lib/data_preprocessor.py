"""Ensure data cleaning, reduction and transformation before classification."""

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from .utils.path import PATH, PREP_METHOD
from .utils.attribute_specifications import ATTRIBUTES, DATA_LABELS


class DataPreprocessor:
    """Preprocess the data."""

    def __init__(self, data):
        """Initialize the data preprocessing."""
        self.data = data

    def preprocess(self):
        """Preprocess the data.

        Returns:
            :return dataFrame: the preprocessed dataset.
        """
        print('Preprocessing the data...')
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
        """Reduce the data."""
        print('Reducing the data...')

        # Remove columns before 'self_valence' column
        sv_index = next((i for i, v in ATTRIBUTES.items() if v == 'self_valence'), None)
        self.data = self.data.iloc[:, sv_index:]

    def normalize_data(self):
        """Transform the data."""
        print('Transforming the data...')

        # Normalize data
        data_columns = [ATTRIBUTES[i] for i in iter(DATA_LABELS)]
        for label in data_columns:
            if PREP_METHOD == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif PREP_METHOD == 'RobustScaler':
                scaler = RobustScaler()
            elif PREP_METHOD == 'StandardScaler':
                scaler = StandardScaler()
            else:
                return
            self.data[label] = scaler.fit_transform(self.data[label].values.reshape(-1, 1))

    def save_preprocessed_data(self):
        """Save the preprocessed data."""
        print('Saving the preprocessed data...')
        self.data.to_csv(PATH['preprocessed_dataset'], index=False)
