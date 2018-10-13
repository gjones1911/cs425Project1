import DataCleaner as DC
import GDataWorks as GDW
import numpy as np
from matplotlib.pyplot import *

# get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")

# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the vals as a list
baddatdic = GDW.FindColBadData(dataarray, '?')

#print(baddatdic)

# TODO: create a function to use linear regression to replace bad data


# adjust the data by replacing bad data with some value say a string version of 0
adjusteddata = list(GDW.ReplaceBadData(dataarray, baddatdic, str(0)))



# adjust the data by making part of it numerical(floats) and giving it a column to
# stop the operation
adjusteddatafloat = list(GDW.MakeDataFloats(adjusteddata, 8))


# function to use averaging to replace bad data
avgdata = list(GDW.AverageReplacer(adjusteddatafloat.copy(), baddatdic, 0.0))


print('avgdata[32]')
print(avgdata[32])
# get all the mpg data as a list
mpgvals, baddatampg = DC.GetCol(avgdata.copy(), 0)
npmpgvals = np.array([mpgvals])
npmpgvalsmean = np.mean(npmpgvals)

print('1 avgdata[32]')
print(avgdata[32])

# get all the cylinder count data as a list
cylindervals, baddatacylinder = DC.GetCol(avgdata, 1)
npcylindervals = np.array([cylindervals])
npcylindervalsmean = np.mean(npcylindervals)


# get all the displacement data as a list
displacementvals, baddatadisplacement = DC.GetCol(avgdata, 2)
npdisplacementvals = np.array([displacementvals])
npdisplacementvalsmean = np.mean(npdisplacementvals)


# get all the horsepower data as a list
horsepowervals, baddatahorsepower = DC.GetCol(avgdata, 3)
nphorsepowervals = np.array([horsepowervals])
nphorsepowervalsmean = np.mean(nphorsepowervals)


# get all the  weight data as a list
weightvals, baddataweight = DC.GetCol(avgdata, 4)
npweightvals = np.array([weightvals])
npweightvalsmean = np.mean(npweightvals)


# get all the acceleration  data as a list
accelerationvals, baddataacceleration = DC.GetCol(avgdata, 5)
npaccelerationvals = np.array([accelerationvals])
npaccelerationvalsmean = np.mean(npaccelerationvals)


# get all the model year  data as a list
modelyearvals, baddatamodelyear = DC.GetCol(avgdata, 6)
npmodelyearvals = np.array([modelyearvals])
npmodelyearvalsmean = np.mean(npmodelyearvals)


# get all the origin data as a list
originvals, baddataorigin = DC.GetCol(avgdata, 7)
nporiginvals = np.array([originvals])
nporiginvalsmean = np.mean(nporiginvals)

print('a avgdata[32]')
print(avgdata[32])
'''
"acceleration": 5,
"model year": 6,
"origin": 7,
"car name": 8}
'''


test = [1,2,3]
'''
#print(sum(test))
print('mpg mean: ' + str(npmpgvalsmean) )
print('clinders mean: ' + str(npcylindervalsmean) )
print('displacment mean: ' + str(npdisplacementvalsmean))
print('horsepower mean: ' + str(nphorsepowervalsmean))
print('weight mean: ' + str(npweightvalsmean))
print('acceleration mean: ' + str(npaccelerationvalsmean))
print('model year  mean: ' + str(np.around(npmodelyearvalsmean)))
print('origin  mean: ' + str(nporiginvalsmean))
'''

print('b avgdata[32]')
print(avgdata[32])
#coef, X, Y = GDW.LinearReggressor(mpgvals, horsepowervals, baddatdic[3])

#coefP2, XP2, YP2 = GDW.P2Reggressor(mpgvals, horsepowervals, baddatdic[3])

#get the m and b using linear regression         X            Y
m, b, Xless, Yless, Yg = GDW.GRegLinRegression(mpgvals, horsepowervals, baddatdic[3])

print('c avgdata[32]')
print(avgdata[32])
# fix missing data points in horse power using linear regression
HPlinReg = GDW.LinRegReplacer(m, b, mpgvals, horsepowervals, baddatdic[3])

print('d avgdata[32]')
print(avgdata[32])

oldavg = avgdata[:]
linadjdata = GDW.LinearReplacer(adjusteddatafloat[:], m, b, 0, 3, baddatdic)

print('2 oldavg[32]')
print(oldavg[32])
Xavg, Yavg = GDW.MakeXYarrays(avgdata, 0, 1, 8)
XLR, YLR = GDW.MakeXYarrays(linadjdata, 0, 1, 8)

print(baddatdic[3])

print(Xavg[32])
print(Yavg)
print(YLR)
print(XLR[32])

# print(W)


#Ynew = GDW.GetYvals(coef,X)
#YP2 = GDW.GetYvals(coefP2,X)

#print(Yg)
figure(1)
plot(Xless, Yless, 'o')
plot(Xless, Yg, 'r--')
# plot(X, np.polyval(coef, X), 'r-')
# plot(XP2, -np.polyval(coefP2,XP2), 'r-')
figure(2)

plot(mpgvals, HPlinReg, 'go')



show()