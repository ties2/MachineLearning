import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    """
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            # Convert each row to float and append to results
            results.append([float(num) for num in row])
    
    # Convert list to numpy array for easier slicing
    results = np.array(results)
    
    # plot precision-recall curve
    plt.plot(results[:, 0], results[:, 1])  # First column as X (precision), second as Y (recall)
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.show()

# Example of using the function
f = open("data_file.csv", "w")
w = csv.writer(f)
_ = w.writerow(["precision", "recall"])
w.writerows([
    [0.013,0.951],
    [0.376,0.851],
    [0.441,0.839],
    [0.570,0.758],
    [0.635,0.674],
    [0.721,0.604],
    [0.837,0.531],
    [0.860,0.453],
    [0.962,0.348],
    [0.982,0.273],
    [1.0,0.0]
])
f.close()

# Plot the data from CSV
plot_data('data_file.csv')

# The problem in plot_data function arises due to a couple of main issues with the data handling and plotting:

# Data Type Issue: When reading from a CSV file using the csv.reader, each row is read as a list of strings. 
# This means when you plot these values, they are treated as categorical data (strings) rather than numerical values, 
# which distorts the plot.

# Incorrect Axis Assignment: Your comment indicates that precision should be on the x-axis and recall on the y-axis, 
# but the plot command actually plots recall on the x-axis and precision on the y-axis, which is inverted.
# Convert String to Float: Each numeric string from the CSV is converted to a float. 
# This is crucial for accurate numerical plotting.
# Correct Axis Plotting: Fixed the plotting to use precision as x-axis and recall as y-axis, as specified.
# This corrected function will now read the numeric data correctly and plot the precision and 
# recall values on the appropriate axes, producing a correct precision-recall curve as expected.