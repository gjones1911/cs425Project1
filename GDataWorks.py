import numpy as np
from scipy.interpolate import *
from DataCleaner import *


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


    independentdata = GetColumnFloat(data, attribs['mpg'])
    dependentdata = GetColumnFloat(data, attribs['horsepower'])

    id = np.array(independentdata)
    dd = np.array(dependentdata)

    p1 = np.polyfit(dd, id, 1)

    print(p1)

    #find bad data if any
    #badDic, rowlist = FindBadDataPoints(data, sig)

    #get the colums you need to focus on


    return fixedMLR

