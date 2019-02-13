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


def view_data_segments(xs, ys, clist, segments):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    fig, ax = plt.subplots()
    colour = np.concatenate([[i] * 20 for i in range(segments)])
    plt.set_cmap('Dark2')
    ax.scatter(xs, ys, c=colour)
    for i in range(segments):
        coefficients = clist[i]
        #Case for last segment (to avoid going out of bounds)
        if (i==segments-1):
            x = np.linspace(xs[20*i], xs[20*(i+1)-1], 100)
        else:
            x = np.linspace(xs[20*i], xs[20*(i+1)], 100)

        y = 0 * x
        for c in coefficients:
            y += c * (x ** coefficients.index(c))
        ax.plot(x, y, linestyle='solid')
    plt.show()


#Calculate the least squares regression for polynomail of degree p+1
def lsrForDegreeD(xs, ys, d):
    columns = []
    for i in range(0,d+1): #can be put inside a list thingy
        columns.append(list(map(lambda x: x**i, xs)))
    XT = np.matrix(columns)
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    #https://stackoverflow.com/a/952952
    coefficients = [item for sublist in A.tolist() for item in sublist]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = 0
        for degree in range(0, d+1):
            yp += coefficients[degree] * (xs[i] ** degree)
        errori = (yp - ys[i])
        sserror += errori ** 2
    return coefficients, sserror

#Try many degree polynomials, use one with least error
def findLsrForBestPolynomial(xs, ys):
    maxdegree = 10
    clist = []
    errorlist = []
    for degree in range(1, maxdegree+1):
        coefficients, error = lsrForDegreeD(xs, ys, degree)
        clist.append(coefficients)
        errorlist.append(error)
    minerror = min(errorlist)
    index = errorlist.index(minerror)
    bestcoefficients = clist[index]
    return bestcoefficients, minerror


def linearlsr(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),xs])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    a = A[0,0]
    b = A[1,0]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * xs[i])
        errori = (yp - ys[i])
        sserror += errori ** 2
    return a, b, sserror

def main(argv):
    totalerror = 0
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
    clist = []
    #Send xs, ys to calculatelsr()
    for i in range(segments):
        coefficients, sserror = findLsrForBestPolynomial(segmentedxs[i], segmentedys[i])
        clist.append(coefficients)
        totalerror += sserror
    print(totalerror)
    if (len(argv) == 3 and argv[2]=="--plot"):
        #do plot stuff
        view_data_segments(xs, ys, clist, segments)


if __name__ == "__main__":
    main(sys.argv)


#Notes: Does noise mean outliers?
