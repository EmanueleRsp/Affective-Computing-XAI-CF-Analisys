"""
This module uses the cfnow library to generate and plot 
counterfactual explanations for a trained model.

The cfnow library is a powerful tool for interpreting machine learning models. 
It provides methods for generating counterfactual explanations, which can help 
understand how a model makes its predictions. This module uses these capabilities 
to generate and plot counterfactual explanations for a trained model on a random graph.
"""
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lib.timer import Timer
from params.path import DIR

# Start timing
timer = Timer()
timer.start()

# Sample 20 random index from dataset
SAMPLES = [1, 9, 50, 70, 99, 127, 1000, 1100, 1300, 1910, 1916,
           1919, 1925, 7711, 19840, 27309, 29971, 33962, 39082, 45981]
print(f"Data sampling of: {SAMPLES}")

# Dir names
source_dir_g = os.path.join(
        DIR['explanation'],
        'greedy'
)
source_dir_c = os.path.join(
        DIR['explanation'],
        'counterShapley'
)
destination_dir_g = os.path.join(
      DIR['explanation'],
      'sampled',
      'greedy'
)
destination_dir_c = os.path.join(
      DIR['explanation'],
      'sampled',
      'counterShapley'
)

# Delete the directories if they exist
shutil.rmtree(destination_dir_g, ignore_errors=True)
shutil.rmtree(destination_dir_c, ignore_errors=True)

# Create directories to store the sample images
os.makedirs(destination_dir_g, exist_ok=True)
os.makedirs(destination_dir_c, exist_ok=True)

for sample in SAMPLES:
    print(f"Copying sample {sample}...")

    # Greedy
    source_file_g = os.path.join(source_dir_g, f'{sample}_greedy_counterplot.png')
    destination_file_g = os.path.join(destination_dir_g, f'{sample}_greedy_counterplot.png')

    try:
        img = mpimg.imread(source_file_g)
    except FileNotFoundError:
        print(f'No greedy plot found for sample {sample}')
        continue

    # Create a new figure with the same size as the image
    DPI = 80
    height, width, _ = img.shape
    figsize = width / float(DPI), height / float(DPI)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Greedy counterplot for sample {sample}")
    plt.subplots_adjust(top=0.8, left=0.1, right=0.9)
    plt.savefig(destination_file_g, dpi=DPI)
    plt.close()

    # CounterShapley
    source_file_c = os.path.join(source_dir_c, f'{sample}_counterShapley_counterplot.png')
    destination_file_c = os.path.join(destination_dir_c, f'{sample}_counterShapley_counterplot.png')
    try:
        img = mpimg.imread(source_file_c)
    except FileNotFoundError:
        print(f'No countershapley plot found for sample {sample}')
        continue

    # Create a new figure with the same size as the image
    DPI = 80
    height, width, _ = img.shape
    figsize = width / float(DPI), height / float(DPI)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Countershapley for sample {sample}")
    plt.subplots_adjust(top=0.8, left=0.1, right=0.9)
    plt.savefig(destination_file_c, dpi=DPI)
    plt.close()

# End timing
timer.end()
