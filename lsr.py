from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def calculatelsr(xs, ys):
    #do calculations
    print("unimplimented")

def main(argv):
    if (len(argv) > 3 or len(argv)==0 ):
        print("Error. Incorrect format.")
        exit()
    if (not os.path.isfile(argv[1])):
        print("Error. File does not exist.")
        exit()

    #Get xs, ys from file specified in command line call
    xs, ys = load_points_from_file(argv[1])
    #Send xs, ys to calculatelsr()
    calculatelsr(xs, ys)

    if (len(argv) == 3 and argv[2]=="--plot"):
        print("Plot detected")
        #do plot stuff


if __name__ == "__main__":
    main(sys.argv)
