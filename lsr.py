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


def view_data_segments(xs, ys, functions):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    segments = len(functions)
    fig, ax = plt.subplots()
    colour = np.concatenate([[i] * 20 for i in range(segments)])
    plt.set_cmap('Dark2')
    ax.scatter(xs, ys, c=colour)

    for function in functions:
        i = functions.index(function)
        #Case for last segment (to avoid going out of bounds)
        if (i==segments-1):
            x = np.linspace(xs[20*i], xs[20*(i+1)-1], 100)
        else:
            x = np.linspace(xs[20*i], xs[20*(i+1)], 100)
        if function[0] == 'exp':
            a = function[2][0]
            b = function[2][1]
            y = a + (b * np.exp(x))
        if function[0] == 'sqrt':
            a = function[2][0]
            b = function[2][1]
            y = a + (b * np.sqrt(x))
        if function[0] == 'inv':
            a = function[2][0]
            b = function[2][1]
            y = a + (b * 1/x)
        if function[0] == 'sin':
            a = function[2][0]
            b = function[2][1]
            y = a + (b * np.sin(x))
        if function[0] == 'tan':
            a = function[2][0]
            b = function[2][1]
            y = a + (b * np.tan(x))
        if isinstance(function[0], int):
            y = 0 * x
            coefficients = function[2]
            for c in coefficients:
                y += c * (x ** coefficients.index(c))
        ax.plot(x, y, linestyle='solid')
    plt.show()

def getCoefficientsFromMatrix(M):
    coefficients = [item for sublist in M.tolist() for item in sublist]
    return coefficients

def lsrForExp(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),np.exp(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * np.exp(xs[i]))
        errori = (yp - ys[i])
        try:
            sserror += errori ** 2
        except Exception:
            pass
    return 'exp', sserror, coefficients

def lsrForSqrt(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),np.sqrt(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * np.sqrt(xs[i]))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'sqrt', sserror, coefficients

def lsrForReciprocal(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),np.reciprocal(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * (np.reciprocal(xs[i])))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'inv', sserror, coefficients

def lsrForSin(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),np.sin(xs)])
    print(XT)
    print(XT.shape)
    print(XT.dtype)
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * np.sin(xs[i]))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'sin', sserror, coefficients


def lsrForTan(xs, ys):
    #do calculations
    XT = np.matrix([np.ones(20),np.tan(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = a + (b * np.tan(xs[i]))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'tan', sserror, coefficients


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
    coefficients = getCoefficientsFromMatrix(A)
    sserror = 0 #Sum squared error
    for i in range(20):
        yp = 0
        for degree in range(0, d+1):
            yp += coefficients[degree] * (xs[i] ** degree)
        errori = (yp - ys[i])
        sserror += errori ** 2
    return degree, sserror, coefficients

#Try many degree polynomials, use one with least error
def findLsrForBestPolynomial(xs, ys):
    maxdegree = 4
    clist = []
    errorlist = []
    tupleList = [] # List of tuples of the form (degree, error, coefficients)
    #For n degree polynomials
    for degree in range(1, maxdegree+1):
        tupleList.append( lsrForDegreeD(xs, ys, degree) )
    #For other functions
    tupleList.append( lsrForExp(xs,ys) )
    tupleList.append( lsrForSqrt(xs, ys) )
    tupleList.append( lsrForReciprocal(xs, ys) )
    tupleList.append( lsrForSin(xs, ys) )
    tupleList.append( lsrForTan(xs, ys) )
    print(tupleList)
    minLine = min(tupleList, key=lambda x: x[1])
    minError = minLine[1]
    coefficientsOfMin = minLine[2]
    minType = [3]
    return minLine


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
    np.seterr(all='ignore')
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
    segmentedxs, segmentedys, numSegments = getlinesegments(xs, ys)
    functions = []
    #Send xs, ys to calculatelsr()
    for i in range(numSegments):
        function = findLsrForBestPolynomial(segmentedxs[i], segmentedys[i])
        print(function[0])
        functions.append(function)
        totalerror += function[1]
    print(totalerror)
    if (len(argv) == 3 and argv[2]=="--plot"):
        #do plot stuff
        view_data_segments(xs, ys, functions)


if __name__ == "__main__":
    main(sys.argv)
