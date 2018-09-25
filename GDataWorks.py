import numpy as np

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


def GDataSpliter(data, splitVal, inc):

    test = []
    verify = []
    validate = []

    Ytest = []
    Yvali = []
    Yver = []

    #get how much data you have
    datacount = len(data)

    splitcount = int(datacount/16) + inc

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

    verend = int(splitcount + splitcount/2)

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



def GrabData(data, startrow, stoprow, startcol, stopcol):

    retdata = []

    for row in range(startrow, stoprow):

        list = []
        for col in range(startcol, stopcol):
            list.append(float(data[row][col]))

        retdata.append(list)
    return retdata

def Gmake(data):



    return

def GmakeMeanArray(data):

    meanArray = []

    for col in range(0,8):
        sum = 0
        for row in range(len(data)):
            sum += float(data[row][col])

        mean = sum/len(data)
        meanArray.append( (sum/(len(data))) )

    return meanArray

def GMulLinReg(indData, depData):

    X = np.array(indData, dtype=np.float64)
    Y = np.array(depData, dtype=np.float64)
    Xtran = np.transpose(X)
    XtranX = np.dot(Xtran, X)
    XTXinv = np.linalg.inv(XtranX)
    XTXinvXT = np.dot(XTXinv, Xtran)
    W = np.dot(XTXinvXT, Y)

    return W

def TestModel(Xdata, Wparams):

    ansArray = []

    for row in Xdata:
        ansArray.append(np.dot(row, Wparams))

    return np.array(ansArray, dtype=np.float64)

def CalculateMSE(Gmodel, Rvalidate):


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
    RSE = 0


    while (datasize/2 + inc < 350):
        test, verify, validate, Ytest, Yver, Yvali = GDataSpliter(DataArray, 1, inc)

        # perform Multiple linear regression
        WparamsT = GMulLinReg(test, Ytest)

        GdataVer = TestModel(verify, WparamsT)

        RSEnew = CalculateMSE(GdataVer, np.array(Yver, dtype=np.float64))
        print("new " + str(RSEnew) )
        if RSEnew > RSE:
            RSE = RSEnew
            bestinc = inc
            print("a better count is " + str(inc))

        inc +=1


        splitcount = int(datasize / 2) + inc

        verend = int(splitcount + splitcount / 2)
        if verend >= datasize:
            print("too big at " + str(verend))
            break

    return RSE  ,bestinc