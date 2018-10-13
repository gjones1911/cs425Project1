import numpy as np
from scipy.interpolate import *
#from DataCleaner import *


#will split the data into 3 vectors
#sample, verify, test
def DataSpliter(data, splitVal, type):

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

    return retvect


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


def data_splitter(xdata, ydata, splitval):

    training_data = []
    validation_data = []

    y_training = []
    y_validation = []

    # grab training data set
    for idx in range(0, splitval):
        training_data.append(xdata[idx])
        y_training.append(ydata[idx])

    # grab validation data set
    for idx in range(splitval, len(xdata)):
        validation_data.append(xdata[idx])
        y_validation.append(ydata[idx])

    return training_data, validation_data, y_training, y_validation



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



# will make an array containing averages of each
# column of data as an entry. Should be given a fixed data set
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

    for idx in range(len(Gmodel)):
        top += np.power((Rvalidate[idx] - Gmodel[idx]), 2)


    for idx in range(len(Rvalidate)):
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


def FixBadDataMLR(data):

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


# finds bad data points and returns a map
# keyed on the column and with a value of a list of the
# rows where the bad data is located
def FindColBadData(dataarray, badsig):

    retdic = {}


    for col in range(len(dataarray[0])):

            for row in range(len(dataarray)):

                if dataarray[row][col] == badsig:

                    if col in retdic:
                        retdic[col].append(row)
                    else:
                        rowlist= [row]
                        retdic[col] = rowlist

    return retdic

# makes the data from col 0 to stop float values
def MakeDataFloats(data, stop):

    array = list(data)

    for row in range(len(array)):
        for col in range(0, stop):
            array[row][col] = float(array[row][col])

    return array


# replaces bad data with the given value
def ReplaceBadData(data, baddic, val):
    for entry in baddic:
        badlist = baddic[entry]
        for idx in range(len(badlist)):
            row = badlist[idx]
            data[row][entry] = val

    return data.copy()


# replaces bad data with the given value
def ReplaceBadDatavec(data, baddic, vec):
    for entry in baddic:
        badlist = baddic[entry]
        for idx in range(len(badlist)):
            row = badlist[idx]
            data[row][entry] = float(vec[idx])

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


        #print('avg:')
        #print(avg)
        #print('entry')
        #print(entry)
        badlist = baddic[entry]

        for row in badlist:
            #print('adding avg: ' + str(avg) + 'to row '+str(row) + ' and col '+ str(entry) )

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


# non mulitvariate
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


def IndeDeparrays(darray, Ycol):

    Y = []
    X = []

    arry = list(darray)

    for row in arry:

        Y.append(row[Ycol])

        #del row[Ycol]

        xc = []
        xc.append(1)

        for idx in range(len(row)-1):
            if idx != Ycol:
                xc.append(row[idx])

        X.append(xc)

    #print(Y)
    #print(X)

    return Y, X


def mulit_linear_regressor(x_data, y_data):

    # dependent array
    # Xarray = []
    # independent array
    # Yarray = []
    # array to work with in the function
    #WorkArry = list(Darray)
    # parameter array
    # W = []
    #print("------------------------------------------------------------------------Work array")
    #print(WorkArry)
    #Yarray, Xarray = IndeDeparrays(list(WorkArry), YCol)
    #print("------------------------------------------------------------------------Work array2")
    #print(WorkArry)
    #Xarry = MakeDataFloats(Xarray,7)

    X = np.array(x_data, dtype=np.float)
    Y = np.array(y_data, dtype=np.float)

    Xtranspose = np.transpose(X)

    XTX = np.dot(Xtranspose, X)

    XTXinv = np.linalg.inv(XTX)

    XTXinvXT = np.dot(XTXinv, Xtranspose)

    W = np.dot(XTXinvXT, Y)

    return W


def MulitLinearRegressor(Darray, YCol):

    # dependent array
    # Xarray = []

    # independent array
    # Yarray = []

    # array to work with in the function
    WorkArry = list(Darray)

    # parameter array
    # W = []

    #print("------------------------------------------------------------------------Work array")
    #print(WorkArry)

    Yarray, Xarray = IndeDeparrays(list(WorkArry), YCol)

    #print("------------------------------------------------------------------------Work array2")
    #print(WorkArry)


    Xarry = MakeDataFloats(Xarray,7)

    X = np.array(Xarry, dtype=np.float)
    Y = np.array(Yarray, dtype=np.float)

    Xtranspose = np.transpose(X)

    XTX = np.dot(Xtranspose, X)

    XTXinv = np.linalg.inv(XTX)

    XTXinvXT = np.dot(XTXinv, Xtranspose)

    W = np.dot(XTXinvXT, Y)

    return W, Xarray, Yarray


def discardbaddata(datalist, badlist):

    count = 0

    workinglist = list(datalist)
    for col in badlist:

        for entry in badlist[col]:
            #print('deleting entry ' + str(entry - count))
            del workinglist[entry - count]
            count += 1

    return workinglist


def samplemeanarray(attribdata):

    end = len(attribdata[0]) - 1

    smu = []

    for col in range(0, end):

        attrib = np.array(GetColumn(attribdata, col), dtype=np.float64)

        smu.append(np.mean(attrib))

    return list(smu)


def get_r_data(x_data, w):

    r = []

    wnp = np.array(w, dtype=np.float64)

    for row in range(len(x_data)):

        x_observation = list()

        #x_observation.append(1)

        for col in range(len(x_data[0])):

            x_observation.append(x_data[row][col])


        r.append(np.dot(np.array(x_observation, dtype=np.float64), wnp))

    return r


def getlinregmissingdata(regdata, baddic, w):

    r = []

    #print('----------------------------------------------------------------regdata')
    #print(regdata)

    for entry in baddic:

        dlist = baddic[entry]


        for row in dlist:
            #print('------------row')
            #print(row)
            x = []
            x.append(1)
            for col in range(len(regdata[0])-1):
                if col != entry:
                    x.append(regdata[row][col])
            Xnp = np.array(x, dtype=np.float64)
            Wnp = np.array(w, dtype=np.float64)

            '''
            print('length of Wnp')
            print(Wnp)
            print('length Xnp')
            print(Xnp)
            '''
            r.append(np.dot(Xnp, Wnp))

    return r


def TrainModel(Xdata, Ydata):

    #training_data, validation_data, y_training, y_validation

    print(len(Xdata))
    split = int(len(Xdata)/16)

    inc = 1

    best_cod = 0
    bs = 10

    train_set, validation_set, y_training, y_validation = data_splitter(Xdata, Ydata, split)

    while len(validation_set) >= 10:

        print('len(train_set)')
        print(len(train_set))

        print('length of validation_set')
        print(len(validation_set))

        print('length of y_training')
        print(len(y_training))

        print('length of y_validation')
        print(len(y_validation))

        w_params = mulit_linear_regressor(train_set, y_training)

        # use w to get some response data
        gmodel = get_r_data(validation_set, w_params)

        cod = CalculateMSE(gmodel, y_validation)

        ans = 1-cod

        if best_cod < cod and cod >  0 and cod <= 1:
            best_cod = cod
            bs = split

        split +=  inc


        if split > len(Xdata)-20:
            print('split')
            print(split)
            break
        print('split norm')
        print(split)
        train_set, validation_set, y_training, y_validation = data_splitter(Xdata, Ydata, split)




    return best_cod, bs
