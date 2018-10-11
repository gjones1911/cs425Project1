import numpy as np
from scipy.interpolate import *
#from DataCleaner import *


#will split the data into 3 vectors
#sample, verify, test
def DataSpliter(data, splitVal, type ):

    test = []
    verify = []
    validate = []

    Ytest = []
    Yvali = []
    Yver = []

    #get how much data you have
    datacount = len(data)

    splitcount = int(datacount/3)

    print("data split count is ",splitcount )

    current = 0

    for idx in range(0, splitcount):

        list = []
        Ytest.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        #print(list)

        test.append(list)


    for idx in range(splitcount, splitcount*2):

        list = []
        Yver.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        verify.append(list)

    for idx in range(splitcount*2, datacount):

        list = []
        Yvali.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        validate.append(list)

    return test, verify, validate, Ytest, Yver, Yvali



def SplitData(data, split):

    test = []


    return


def GetPart(vector, exclude):

    retvect = []

    for i in range(len(vector)):
        if i != exclude:
            retvect.append(vector[i])

    return  retvect

def GetColumn(array2d, col):

    retlist = []
    for entry in array2d:

        item = entry[col]

        retlist.append(item)

    return retlist


def GetColumnFloat(array2d, col):

    retlist = []
    for entry in array2d:

        item = float(entry[col])

        retlist.append(item)

    return retlist


def GetGrouping2Dremove(list, rs, re, cs, ce, rmv):

    retlist = []

    for r in range(rs, re+1):
        row = list[r][cs:ce+1]

        row = GetPart(row,rmv)

        retlist.append(row)

    return retlist



def GetGrouping2D(list, rs, re, cs, ce):

    retlist = []

    for r in range(rs, re+1):
        row = list[r][cs:ce+1]

        retlist.append(row)

    return retlist



def GDataSpliter(data, splitVal, inc):

    test = []
    verify = []
    validate = []

    Ytest = []
    Yvali = []
    Yver = []

    #get how much data you have
    datacount = len(data)

    splitcount = int(datacount/splitVal) + inc

    left_ovr = datacount-splitcount

    #print("data split count is ",splitcount )

    current = 0

    #grab test data
    for idx in range(0, splitcount):

        list = []
        Ytest.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        #print(list)

        test.append(list)
    #####################################################33

    verend = int(splitcount + splitcount/2)-5
    #verend = int(splitcount + left_ovr/2)

    #grab verification data
    for idx in range(splitcount, verend):
        #print(idx)
        list = []
        Yver.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        verify.append(list)
    ###########################################################

    #grab validation data

    for idx in range(verend, datacount):

        list = []
        Yvali.append(data[idx][0])
        list.append(1.0)
        for col in range(1,8):
            attr = float( data[idx][col] )
            list.append(attr)

        validate.append(list)

    return test, verify, validate, Ytest, Yver, Yvali


#grabs a section of data of data from given data array
def GrabData(data, startrow, stoprow, startcol, stopcol):

    retdata = []

    for row in range(startrow, stoprow):

        list = []
        for col in range(startcol, stopcol):
            list.append(float(data[row][col]))

        retdata.append(list)
    return retdata


#will make an array containing averages of each
#column of data as an entry. Should be given a fixed
#data set
def GmakeMeanArray(data):

    meanArray = []

    #go through each column row by row
    #summing entries and getting averages
    for col in range(0,8):
        sum = 0
        for row in range(len(data)):
            sum += float(data[row][col])

        meanArray.append( (sum/(len(data))) )

    return meanArray


def GMulLinReg(indData, depData):

    #make numpy arrays
    X = np.array(indData, dtype=np.float64)
    Y = np.array(depData, dtype=np.float64)

    #get the transpose for the calculation
    Xtran = np.transpose(X)

    #calculate  Xtranpose * X
    XtranX = np.dot(Xtran, X)


    #calculate  get the inverse
    XTXinv = np.linalg.inv(XtranX)

    #calculate Xtranpose*X_inverse * Xtranspose
    XTXinvXT = np.dot(XTXinv, Xtran)

    #calculate the W(parameter) vector
    W = np.dot(XTXinvXT, Y)

    return W

#will use the independent data(Xdata) and the given params(Wparams)
#and return an dependent(Y) array
def TestModel(Xdata, Wparams):

    ansArray = []

    for row in Xdata:
        ansArray.append(np.dot(row, Wparams))

    return np.array(ansArray, dtype=np.float64)


#calculates the Mean square error of the model using the validation data
def CalculateMSE(Gmodel, Rvalidate):

    #get the mean of the validation data
    Rmean = np.mean(Rvalidate)
    bottom = 0
    top = 0

    for idx in range(Gmodel.size):
        top += np.power((Rvalidate[idx] - Gmodel[idx]), 2)


    for idx in range(Rvalidate.size):
        bottom += np.power((Rvalidate[idx] - Rmean), 2)

    TpDivBtm = top/bottom

    return 1 - TpDivBtm


def GTrainer(DataArray):

    inc = 0

    datasize = len(DataArray)

    bestinc = 0
    bestinc2 = 0
    RSE = 0

    RSE2 = 0
    bestTestSize = 0
    bestTestSize2 = 0

    splitval = 16

    #Yvali = []

    while ( int(datasize/splitval) + inc < 398):

        testsize = int(datasize/splitval) + inc


        left_ovr = datasize - testsize

        #split the data into parts
        test, verify, validate, Ytest, Yver, Yvali = GDataSpliter(DataArray, splitval, inc)

        # perform Multiple linear regression
        WparamsT = GMulLinReg(test, Ytest)

        GdataVer = TestModel(verify, WparamsT)

        RSEnew = CalculateMSE(GdataVer, np.array(Yver, dtype=np.float64))
        print("new ver " + str(RSEnew) )


        Gdatavall = TestModel(validate, WparamsT)
        RSEvall = CalculateMSE(Gdatavall, np.array(Yvali, dtype=np.float64))
        print("new vall" + str(RSEvall) )


        #if a better error marker is found save the parameters that
        #got to it
        if RSEnew > RSE:
            RSE = RSEnew
            bestinc = inc
            bestTestSize = testsize
            print("a better count is " + str(inc))

        if RSEvall > RSE2:
            RSE2 = RSEvall
            bestinc2 = inc
            bestTestSize2 = testsize
            print("a better count is " + str(inc))

        inc +=1


        splitcount = int(datasize / splitval) + inc

        verend = int(splitcount + splitcount / 2)
        if verend >= datasize:
            print("too big at " + str(verend))
            break

    print("testL verL valL")
    print(len(test), len(verify), len(validate))

    Gdataval = TestModel(validate, WparamsT)
    RSEval = CalculateMSE(Gdataval, np.array(Yvali, dtype=np.float64))

    print("best vali: " + str(RSE2) +" bestinc2 "+ str(bestinc2)+" besttestsize2 "+ str(bestTestSize2))
    print("the val coefficient of determination: " + str(RSEval))

    return RSE, bestinc, bestTestSize

def FindBadDataPoints(data, sig):

    badDataDic = {}
    collist = []


    r = 0

    for row in data:
        c = 0
        for col in row:

            if col == sig:

                if c not in badDataDic:
                    list = [r]
                    collist.append(c)
                    badDataDic[c] = list
                else:
                    badDataDic[c].append(r)
            c += 1

        r += 1
    return badDataDic, collist

def GetBadColumns(dic):

    retdic = {}

    for item in dic:
        col = dic[item]
        if col not in retdic:
            list = [item]
            retdic[col] = list
        else:
            retdic[col].append(item)
        #print(dic[item])

    return retdic


def FixBadDataMLR(data ):

    fixedMLR = []


    independentdata = GetColumnFloat(data, 0)
    dependentdata = GetColumnFloat(data, 3)

    id = np.array(independentdata)
    dd = np.array(dependentdata)

    p1 = np.polyfit(dd, id, 1)

    print(p1)

    #find bad data if any
    #badDic, rowlist = FindBadDataPoints(data, sig)

    #get the colums you need to focus on


    return fixedMLR


#finds bad data points and returns a map
#keyed on the column and with a value of a list of the
#rows where the bad data is located
def FindColBadData(dataarray):

    retdic = {}

    for col in range(len(dataarray[0])):

            for row in range(len(dataarray)):

                if dataarray[row][col] == '?':

                    if col in retdic:
                        retdic[col].append(row)
                    else:
                        rowlist= [row]
                        retdic[col] = rowlist

    return retdic

#makes the data from col 0 to stop float values
def MakeDataFloats(data, stop):
    for row in range(len(data)):
        for col in range(0, stop):
            data[row][col] = float(data[row][col])

    return data.copy()

#replaces bad data with the given value
def ReplaceBadData(data, baddic, val):
    for entry in baddic:
        badlist = baddic[entry]
        for idx in range(len(badlist)):
            row = badlist[idx]
            data[row][entry] = val

    return data.copy()


def BadDataAverager(data, col, sigval):

    colsum = 0
    count = 0
    for row in range(len(data)):

        val = data[row][col]

        if val != sigval:
            colsum += val
            count += 1

    return colsum/count

def AverageReplacer(data, baddic, sigval):

    for entry in baddic:
        avg = BadDataAverager(data, entry, sigval)


        print('avg:')
        print(avg)
        print('entry')
        print(entry)
        badlist = baddic[entry]

        for row in badlist:
            print('adding avg: ' + str(avg) + 'to row '+str(row) + ' and col '+ str(entry) )

            data[row][entry] = avg

    return data.copy()

def LinearReplacer(data, m, b, Xcol, Ycol, baddataPoints):

    datapointrows = baddataPoints[Ycol]

    for row in datapointrows:
        x = data[row][Xcol]

        data[row][Ycol] = m*x + b

    return data.copy()

def LinearReggressor(independent, dependent, datatoremove):

    for item in range(len(datatoremove)):
        del independent[item]
        del dependent[item]

    print("independent length: ")
    print(len(independent))
    print("dependent length: ")
    print(len(dependent))


    coeff = np.polyfit(independent, dependent, 1)
    return coeff, dependent, independent

def P2Reggressor(independent, dependent, datatoremove):

    for item in range(len(datatoremove)):
        del independent[item]
        del dependent[item]

    print("independent length: ")
    print(len(independent))
    print("dependent length: ")
    print(len(dependent))


    coeff = np.polyfit(independent, dependent, 2)
    return coeff, dependent, independent


def GetYvals(coef, X):

    Y = []

    m = coef[0]
    b = coef[1]

    for idx in range(len(X)):

        val = m*X[idx]+b
        Y.append(val)


    return Y

def GetYvalsP2(coef, X):

    Y = []

    mx2 = coef[0]
    mx = coef[1]
    b = coef[2]

    for idx in range(len(X)):

        val = mx2 * np.power(X[idx],2) + mx*np.power(X[idx]) + b
        Y.append(val)


    return Y



def GRegLinRegression(X, Y, datatoremove):

    for item in range(len(datatoremove)):
        del X[item]
        del Y[item]


    Xsum = sum(X)
    Ysum = sum(Y)

    XY = [a*b for a,b in zip(X, Y)]

    XX = [a*b for a,b in zip(X, X)]

    XXsum = sum(XX)

    XYsum = sum(XY)

    N = len(X)

    A = [[N, Xsum],
         [Xsum, XXsum]]

    y = [[Ysum],
         [XYsum]]

    Anp = np.array([A])

    Anpinv = np.linalg.inv(Anp)

    Ynp = np.array([y])

    W = np.dot(Anpinv, Ynp)

    b = W[0][0][0][0]
    m = W[0][1][0][0]

    print('b: ')
    print(b)
    print('m: ')
    print(m)

    Yg = []

    for idx in range(len(X)):
        val = m*X[idx] + b
        Yg.append(val)



    return m, b, X, Y, Yg

# TODO fix this so it calls the method above and remove m, and b from argument list
def LinRegReplacer(m,b,X, Y, missindatapoints):

    for idx in missindatapoints:

        x = X[idx]

        Y[idx] = x*m + b

        print(Y[idx])

    return Y

def MakeXYarrays(data, Ystart, Xstart, Xstop):

    Y = []
    X = []

    for row in data:
        list0 = row
        Y.append(list0[Ystart])
        X.append(list0[Xstart:Xstop])

    return X, Y

