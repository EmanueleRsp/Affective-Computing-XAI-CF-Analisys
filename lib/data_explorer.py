"""Raw data analysis, to visualize and plot their characteristics."""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lib.utils.path import DIR, PREP_METHOD
from lib.utils.attribute_specifications import (
    ATTRIBUTES, CLASS_LABELS, DATA_LABELS, BRAINWAVE_BANDS
)


class DataExplorer:
    """Explore the data."""

    def __init__(self, data, directory):
        """Initialize the data exploration."""
        self.data = data
        self.directory = directory

    def exploration(self):
        """Explore the data."""

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
        if self.directory == DIR['raw_data_img']:
            for index, label in bw_columns.items():
                self._bw_plot(index, label)

        # Plot other data characteristics
        for index, label in data_columns.items():
            if label in CLASS_LABELS:
                continue
            self._data_plot(index, label)

    def _categorical_plot(self, index, label):
        """Plot categorical data characteristics."""
        print(f'Plotting {label}...')
        index = str(index).zfill(2)

        # Bar plot of the data
        _, axis = plt.subplots()
        self._create_bar(axis, label)
        plt.savefig(os.path.join(self.directory, f'{index}_{label}_bar.png'))
        plt.close()

    def _create_bar(self, axs, label):
        """Create bar plot of the data."""

        counts = self.data[label].value_counts()
        axs.bar(counts.sample, counts.values)
        axs.set_xlabel(label)
        axs.set_ylabel('Count')
        axs.set_title(f'{label} bar plot')
        axs.grid(True)
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _bw_plot(self, index, label):
        """Plot brainwave band characteristics."""
        print(f'Plotting {label} brainwave...')
        index = str(index).zfill(2)

        # Temporal graph of the data
        if self.directory == DIR['raw_data_img']:
            _, axis = plt.subplots()
            self.create_temporal_graph(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {PREP_METHOD}')
            path = os.path.join(DIR['raw_data_img'], f'temporal_graph_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self.create_temporal_graph(axs[1], label)
        plt.savefig(os.path.join(self.directory, f'temporal_graph_{index}_{label}.png'))
        plt.close()

    def create_temporal_graph(self, axs, label):
        """Create temporal graph of the data."""

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
        """Plot data characteristics."""
        print(f'Plotting {label}...')
        index = str(index).zfill(2)

        # Histogram of the data
        if self.directory == DIR['raw_data_img']:
            _, axis = plt.subplots()
            self._create_histogram(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {PREP_METHOD}')
            path = os.path.join(DIR['raw_data_img'], f'histogram_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self._create_histogram(axs[1], label)
        plt.savefig(os.path.join(self.directory, f'histogram_{index}_{label}.png'))
        plt.close()

        # Box plot of the data
        if self.directory == DIR['raw_data_img']:
            _, axis = plt.subplots()
            self._create_box_plot(axis, label)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            fig.suptitle(f'{label} - Before and after {PREP_METHOD}')
            path = os.path.join(DIR['raw_data_img'], f'box_plot_{index}_{label}.png')
            img = mpimg.imread(path)
            axs[0].imshow(img)
            axs[0].axis('off')
            axs[0].set_aspect(aspect='equal')
            box = axs[1].get_position()
            axs[1].set_position([box.x0, box.y0, box.width, box.height - 0.05])
            self._create_box_plot(axs[1], label)
        plt.savefig(os.path.join(self.directory, f'box_plot_{index}_{label}.png'))
        plt.close()

    def _create_histogram(self, axs, label):
        """Create histogram of the data."""

        axs.hist(self.data[label], bins=50)
        axs.set_xlabel(label)
        axs.set_ylabel('Frequency')
        axs.set_title(f'{label} histogram')
        axs.grid(True)

    def _create_box_plot(self, axs, label):
        """Create box plot of the data."""

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
        """Create bar plot of the data."""

        axs.bar(self.data[label], self.data[label])
        axs.set_xlabel(label)
        axs.set_ylabel('Value')
        axs.set_title(f'{label} bar plot')
        axs.grid(True)
