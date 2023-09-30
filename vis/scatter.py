from numpy import array
from matplotlib import pyplot as plt

def plot_array(array_to_plot: array) -> None:
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Random data of 100×3 dimension

    # Scatter plot
    plt.scatter(array_to_plot[:, 0], array_to_plot[:, 1], cmap='hot')

    # Display the plot
    plt.show()