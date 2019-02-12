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

def getlinesegments(xs, ys):
    size = len(xs)
    segmentedxs = [xs[i:i+20] for i in range(0, size, 20)]
    segmentedys = [ys[i:i+20] for i in range(0, size, 20)]
    segments = size//20
    return segmentedxs, segmentedys, segments


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
    XT = np.matrix([np.ones(20),xs])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    print(XT)
    print(X)
    A = ((XT * X).getI() * XT) * Y
    print(A)

def main(argv):
    if (len(argv) > 3 or len(argv)==1 ):
        print("Error. Incorrect format.")
        exit()

    #Check if input file exists
    if (not os.path.isfile(argv[1])):
        print("Error. File does not exist.")
        exit()

    #Get xs, ys from file specified in command line call
    filename = argv[1]
    xs, ys = load_points_from_file(filename)
    segmentedxs, segmentedys, segments = getlinesegments(xs, ys)

    #Send xs, ys to calculatelsr()
    for i in range(segments):
        calculatelsr(segmentedxs[i], segmentedys[i])

    if (len(argv) == 3 and argv[2]=="--plot"):
        print("Plot detected")
        #do plot stuff
        view_data_segments(xs, ys)




if __name__ == "__main__":
    main(sys.argv)
