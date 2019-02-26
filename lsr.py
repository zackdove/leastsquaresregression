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
    """Splits the long lists of xs, ys into a list of lists of size 20
    Args:
        xs : list of x values
        ys : list of y values
    Returns:
        segmentedxs : list of lists of size 20 containing the xs
        segmentedys : list of lists of size 20 containing the ys
        segments : number of segments
        """
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

        if isinstance(function[0], int):
            y = 0 * x
            coefficients = function[2]
            for c in coefficients:
                y += c * (x ** coefficients.index(c))
        else:
            a = function[2][0]
            b = function[2][1]
            if function[0] == 'exp':
                y = a + (b * np.exp(x))
            if function[0] == 'sqrt':
                y = a + (b * np.sqrt(x))
            if function[0] == 'inv':
                y = a + (b * 1/x)
            if function[0] == 'sin':
                y = a + (b * np.sin(x))
            if function[0] == 'tan':
                y = a + (b * np.tan(x))
        ax.plot(x, y, linestyle='solid')
    plt.show()

def getCoefficientsFromMatrix(M):
    """Takes the individul values from a matrix
    Args:
        M : multi dimensional matrix containing a coefficients
    Returns:
        coefficients : a list of coefficients
    """
    coefficients = [item for sublist in M.tolist() for item in sublist]
    return coefficients

def lsrForExp(xs, ys):
    """Calculates the least squares regression for exponential functions of the
    form y=a+b*exp(x), as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        'exp' : the type of function returned
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    XT = np.matrix([np.ones(20),np.exp(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0
    for i in range(20):
        yp = a + (b * np.exp(xs[i]))
        errori = (yp - ys[i])
        try:
            sserror += errori ** 2
        except Exception:
            pass
    return 'exp', sserror, coefficients

def lsrForSqrt(xs, ys):
    """Calculates the least squares regression for square root functions of the
    form y=a+b*sqrt(x), as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        'sqrt' : the type of function returned
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
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
    """Calculates the least squares regression for reciprocal functions of the
    form y=a+b/(x), as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        'inv' : the type of function returned
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    XT = np.matrix([np.ones(20),np.reciprocal(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0
    for i in range(20):
        yp = a + (b * (np.reciprocal(xs[i])))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'inv', sserror, coefficients

def lsrForSin(xs, ys):
    """Calculates the least squares regression for sin functions of the
    form y=a+b*sin(x), as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        'sin' : the type of function returned
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    XT = np.matrix([np.ones(20),np.sin(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0
    for i in range(20):
        yp = a + (b * np.sin(xs[i]))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'sin', sserror, coefficients


def lsrForTan(xs, ys):
    """Calculates the least squares regression for tan functions of the
    form y=a+b*tan(x), as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        'tan' : the type of function returned
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    XT = np.matrix([np.ones(20),np.tan(xs)])
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    a = coefficients[0]
    b = coefficients[1]
    sserror = 0
    for i in range(20):
        yp = a + (b * np.tan(xs[i]))
        errori = (yp - ys[i])
        sserror += errori ** 2
    return 'tan', sserror, coefficients


#Calculate the least squares regression for polynomail of degree d+1
def lsrForDegreeD(xs, ys, d):
    """Calculates the least squares regression for polynomail functions of the
    form y=a+b*x^d, as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
        d : the exponent of x, also the degree-1
    Returns:
        degree : the type of degree of the polynomial
        sserror : the sum squared error to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    columns = []
    for i in range(0,d+1):
        columns.append(list(map(lambda x: x**i, xs)))
    XT = np.matrix(columns)
    X = XT.getT()
    Y = np.matrix(ys).getT()
    A = ((XT * X).getI() * XT) * Y
    coefficients = getCoefficientsFromMatrix(A)
    sserror = 0
    for i in range(20):
        yp = 0
        for degree in range(0, d+1):
            yp += coefficients[degree] * (xs[i] ** degree)
        errori = (yp - ys[i])
        sserror += errori ** 2
    return degree, sserror, coefficients

#Try many degree polynomials, use one with least error
def findLsrForBestFunction(xs, ys):
    """Given a set of xs and ys, finds the least squares reggression line with
    the lowest sum squared error by trying many functions.
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        minLine : a tuple (type, error, coefficients) representing the best
        line
        type : the type of the function, either an integer representing a
        polynomial or a string 'exp', 'sin' etc...
        error : the sum squared error of the line to the data points
        coefficients : a list containing the coefficients for the reggression
        line
    """
    maxdegree = 4
    tupleList = []
    #For n degree polynomials
    for degree in range(1, maxdegree+1):
        tupleList.append( lsrForDegreeD(xs, ys, degree) )
    #For other functions
    tupleList.append( lsrForExp(xs,ys) )
    tupleList.append( lsrForSqrt(xs, ys) )
    tupleList.append( lsrForReciprocal(xs, ys) )
    tupleList.append( lsrForSin(xs, ys) )
    tupleList.append( lsrForTan(xs, ys) )
    minLine = min(tupleList, key=lambda x: x[1])
    minError = minLine[1]
    coefficientsOfMin = minLine[2]
    minType = [3]
    return minLine


def linearlsr(xs, ys):
    """----UNUSED----
    Calculates the least squares regression for linear functions of the
    form y=a+b*x, as well as the error
    Args:
        xs : a list of 20 x values
        ys : a list of 20 y values
    Returns:
        a, b : the coefficients of the regression line
        sserror : the sum squared error to the data points
    """
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
    """Loads data points from a file input, splits into 20 point segments,
    finds and plots the best fitting least squares regression line for each
    segment, and then prints the total sum squared error
    Args:
        argv[1] : the name of the .csv file containing the points
        argv[2] : '--plot' if to plot the lines, or blank if not
    """
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
        function = findLsrForBestFunction(segmentedxs[i], segmentedys[i])
        functions.append(function)
        totalerror += function[1]
    print(totalerror)
    if (len(argv) == 3 and argv[2]=="--plot"):
        view_data_segments(xs, ys, functions)


if __name__ == "__main__":
    main(sys.argv)
