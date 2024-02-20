"""
This module contains the DataExplorer class.

The DataExplorer class is used to visualize and plot the characteristics of a given dataset.
It includes methods for exploring the data, creating bar plots, temporal graphs, histograms,
and box plots for different types of data (categorical, brainwave bands, and other data).

Typical usage example:

    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> explorer = DataExplorer(data, 'save_dir')
    >>> explorer.exploration()
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from params.attribute_specifications import (
    ATTRIBUTES, CLASS_LABELS, DATA_LABELS, BRAINWAVE_BANDS
)


class DataExplorer:
    """Explore the data.

    This class provides methods to explore and visualize data.

    Attributes:
        data (pandas.DataFrame): The data to be explored.
        save_dir (str): The directory to save the exploration results.
        raw_dir (str): raw data directory (if exploring prep data).
            Default None.
        prep_method (str): prep method used (if exploring prep data).
            Default None.
    """

    def __init__(self, data, save_dir, raw_dir=None, prep_method=None):
        """Initialize the data exploration.

        Args:
            data (pandas.DataFrame): The data to be explored.
            save_dir (str): The directory to save the exploration results.
            raw_dir (str): raw data directory (if exploring prep data).
                Default None.
            prep_method (str): prep method used (if exploring prep data).
                Default None.
        """

        self.data = data
        self.save_dir = save_dir
        self.raw_dir = raw_dir
        self.prep_method = prep_method

    def exploration(self):
        """Explore the data.

        This method performs various data exploration tasks, such as checking for null values,
        duplicates, and creating visualizations of the data.
        """

        print(f'Shape: {self.data.shape}')

        print('Searching for null values...')
        if self.data.isnull().values.any():
            print(f'Null values found: {self.data.isnull().sum().sum()}')
        else:
            print('No null values found!')

        print('Searching for duplicates...')
        if self.data.duplicated().any():
            print(f'Duplicates found: {self.data.duplicated().sum()}')
        else:
            print('No duplicates found!')

        # # Number of values occurrences in 'seconds' column
        # print('Seconds column value counts:')
        # print(self.data['seconds'].value_counts())

        print('Creating images...')
        class_columns = {i: ATTRIBUTES[i] for i in iter(CLASS_LABELS)}
        data_columns = {i: ATTRIBUTES[i] for i in iter(DATA_LABELS)}
        bw_columns = {i: ATTRIBUTES[i] for i in iter(BRAINWAVE_BANDS)}

        # Plot categorical data characteristics
        for index, label in class_columns.items():
            self._categorical_plot(index, label)

        # Plot brainwave band characteristics
        if self.raw_dir is None:
            for index, label in bw_columns.items():
                self._bw_plot(index, label)

        # Plot other data characteristics
        for index, label in data_columns.items():
            if label in CLASS_LABELS:
                continue
            self._data_plot(index, label)

    def _categorical_plot(self, index, label):
        """
        Plot categorical data characteristics.

        Parameters:
        index (int): The index of the plot.
        label (str): The label of the plot.

        Returns:
        None
        """
        print(f'Plotting {label}...')
        index = str(index).zfill(2)

        # Bar plot of the data
        _, axis = plt.subplots()
        self._create_bar(axis, label)
        plt.savefig(os.path.join(self.save_dir, f'{index}_{label}_bar.png'))
        plt.close()

    def _create_bar(self, axs, label):
        """
        Create a bar plot of the data.

        Parameters:
            axs (matplotlib.axes.Axes): The axes object to plot on.
            label (str): The label of the data to plot.

        Returns:
            None
        """

        counts = self.data[label].value_counts()
        axs.bar(counts.sample(), counts.values)
        axs.set_xlabel(label)
        axs.set_ylabel('Count')
        axs.set_title(f'{label} bar plot')
        axs.grid(True)
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _bw_plot(self, index, label):
        """
        Plot brainwave band characteristics.

        Parameters:
        index (int): The index of the brainwave band.
        label (str): The label of the brainwave band.

        Returns:
        None
        """

        print(f'Plotting {label} brainwave...')
        index = str(index).zfill(2)

        # Temporal graph of the data
        if self.raw_dir is None:
            _, axis = plt.subplots()
            self.create_temporal_graph(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {self.prep_method}')
            path = os.path.join(self.raw_dir, f'temporal_graph_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self.create_temporal_graph(axs[1], label)
        plt.savefig(os.path.join(self.save_dir, f'temporal_graph_{index}_{label}.png'))
        plt.close()

    def create_temporal_graph(self, axs, label):
        """
            Create a temporal graph of the data.

            Parameters:
            axs (matplotlib.axes.Axes): The axes object to plot the graph on.
            label (str): The label of the data to be plotted on the y-axis.

            Returns:
            None
            """
        times = []
        values = []

        times_count = self.data['seconds'].value_counts().sort_index()
        for time, count in times_count.items():
            for i in range(count):
                times.append((1 - i / count) * time + (i / count) * (time + 5))
                values.append(self.data[self.data['seconds'] == time][label].iloc[i])
        axs.plot(times, values)
        axs.set_xlabel('Time (seconds)')
        axs.set_ylabel(label)
        axs.set_title(f'{label} temporal graph')
        axs.grid(True)

    def _data_plot(self, index, label):
        """
            Plot data characteristics.

            Parameters:
            index (int): The index of the data.
            label (str): The label of the data.

            Returns:
            None
            """
        print(f'Plotting {label}...')
        index = str(index).zfill(2)

        # Histogram of the data
        if self.raw_dir is None:
            _, axis = plt.subplots()
            self._create_histogram(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {self.prep_method}')
            path = os.path.join(self.raw_dir, f'histogram_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self._create_histogram(axs[1], label)
        plt.savefig(os.path.join(self.save_dir, f'histogram_{index}_{label}.png'))
        plt.close()

        # Box plot of the data
        if self.raw_dir is None:
            _, axis = plt.subplots()
            self._create_box_plot(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {self.prep_method}')
            path = os.path.join(self.raw_dir, f'box_plot_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self._create_box_plot(axs[1], label)
        plt.savefig(os.path.join(self.save_dir, f'box_plot_{index}_{label}.png'))
        plt.close()

    def _create_histogram(self, axs, label):
        """
        Create a histogram of the data.

        Parameters:
        axs (matplotlib.axes.Axes): The axes object to plot the histogram on.
        label (str): The label of the data to plot.

        Returns:
        None
        """

        axs.hist(self.data[label], bins=50)
        axs.set_xlabel(label)
        axs.set_ylabel('Frequency')
        axs.set_title(f'{label} histogram')
        axs.grid(True)

    def _create_box_plot(self, axs, label):
        """Create box plot of the data.

            Args:
                axs (matplotlib.axes.Axes): The axes object to plot the box plot on.
                label (str): The label of the data to create the box plot for.

            Returns:
                None
            """

        # Create the box plot
        box_plot = axs.boxplot(self.data[label])
        # Add annotations for the whisker values
        lower_whisker = box_plot['whiskers'][0].get_ydata()[1]
        upper_whisker = box_plot['whiskers'][1].get_ydata()[1]
        axs.annotate(f'Lower: {lower_whisker:.2f}', (1, lower_whisker),
                     textcoords="offset points", xytext=(-70, -10), ha='center')
        axs.annotate(f'Upper: {upper_whisker:.2f}', (1, upper_whisker),
                     textcoords="offset points", xytext=(-70, 10), ha='center')
        axs.set_xlabel(label)
        axs.set_ylabel('Frequency')
        axs.set_title(f'{label} box plot')
        axs.grid(True)

    def create_bar_plot(self, axs, label):
        """
        Create a bar plot of the data.

        Parameters:
            axs (matplotlib.axes.Axes): The axes object to plot on.
            label (str): The label of the data to plot.

        Returns:
            None
        """

        axs.bar(self.data[label], self.data[label])
        axs.set_xlabel(label)
        axs.set_ylabel('Value')
        axs.set_title(f'{label} bar plot')
        axs.grid(True)
