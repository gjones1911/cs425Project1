'''
Created by: Gerald Jones
Purpose: This will hopefully use linear regression to model how
         the number of cylinders, displacement, horsepower, weight,
         acceleration, model year, and origin of a car can be used with
         linear regression techniques to predict mpg. This one uses linear
         regression to replace missing data
'''

import numpy as np
from matplotlib.pyplot import *
import GDataWorks as GDW
import DataCleaner as DC

# get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")

error_array = ['COD', 'Least_squares', 'MSE']

#print('data0')
#print(dataarray[32])
# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the vals as a list
baddatdic = GDW.FindColBadData(dataarray.copy(), '?')

print(baddatdic)

#print('data1')
#print(dataarray[32])
# adjust the data by replacing bad data with some value say a string version of 0

adjusteddata = list(GDW.ReplaceBadData(list(dataarray), baddatdic, str(0)))

#print('------------------------------------------------------------------------------------adjusteddata')
#print(adjusteddata)


adjusteddata = GDW.discardbaddata(list(adjusteddata), baddatdic)


#print('------------------------------------------------------------------------------------adjusteddata2')
#print(adjusteddata)


#print('data2')
#print(dataarray[32])

origdata = list(dataarray)


#print('------------------------------------------------------------------------------------origdataA')
#print(origdata)


# adjust the data by making part of it numerical(floats) and giving it a column to
# stop the operation
adjusteddatafloat = GDW.MakeDataFloats(list(adjusteddata), 8)
origdata = GDW.MakeDataFloats(list(origdata), 8)

#print('------------------------------------------------------------------------------------origdataB')
#print(origdata)



Whp, Xxmd, Yymd = GDW.MulitLinearRegressor(adjusteddatafloat, 3)
print('Whp')
print(Whp)

#print('/n')
#print('/n')
#print('------------------------------------------------------------------------------------origdataC')
#print(origdata)



rt = GDW.getlinregmissingdata(list(origdata), baddatdic, Whp)
#print('-------------------------------------rt')
#print(rt)

#print("adjusted floats vs. original floats")
#print(adjusteddatafloat[32])
#print(origdata[32])
#print(origdata[1])

adjustedorigdata = GDW.ReplaceBadDatavec(origdata, baddatdic, rt)

print('adjustedorigdata[32]')
print(adjustedorigdata[32])

#print('+++++++++++++++++++++++++++++++-----------------------------adjusted original data after replacement')
#print('adjustedorigdata[32]')
#print(adjustedorigdata[32])
#print('adjustedorigdata[1]')
#print(adjustedorigdata[1])
'''
print('original data')
print(origdata[32])
print('adjusted data')
print(adjusteddata[32])
print('adjusteddatafloat')
print(adjusteddatafloat[32])
'''
# function to use averaging to replace bad data
#avgdata = list(GDW.AverageReplacer(adjusteddatafloat.copy(), baddatdic, 0.0))

# get all the mpg data as a list
mpgvals, baddatampg = DC.GetCol(adjustedorigdata.copy(), 0)
npmpgvals = np.array([mpgvals])
npmpgvalsmean = np.mean(npmpgvals)

print('mpg mean')
print(npmpgvalsmean)

smu = GDW.samplemeanarray(adjustedorigdata.copy())

print('smu')
print(smu)

Wmpg, Xx, Yy = GDW.MulitLinearRegressor(adjustedorigdata, 0)

print('Wmpg')
print(len(Wmpg))
print(Wmpg)

print('adjusted original data as float')
print(len(adjustedorigdata))

print('X')
print(len(Xx))
print(len(Xx[0]))
print(Xx)

print('Y')
print(len(Yy))
print(1)
print(Yy)

# Coefficient of Determination :
best_cod, bs = GDW.TrainModelCOD(Xx, Yy)
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs)

# Lest Squares Error :
best_ls, bs = GDW.TrainModelLSE(Xx, Yy)
print('best Least Squares erro')
print(best_ls)
print('best split Least Squares error')
print(bs)

# Meas Square Error :
best_mse, bs = GDW.TrainModelMSE(Xx, Yy)
print('best MSE')
print(best_mse)
print('best split MSE')
print(bs)


F, best_mse = GDW.forward_selector(Xx, Yy, bs)

print('best MSR is ')
print(best_mse)
print('F')
print(F)


