#import DataCleaner as DC
import numpy as np
#import GDataWorks as GDW

x = [2, 3, 4, 10, 5]
y = [2, 1, 3, 5, 8]



xy = [a*b for a, b in zip(x, y)]

alllist = [[x],
           [y],
           [xy]]

newX = alllist[:]

prt = []
y = []

list0 = alllist[0][0]
list1 = alllist[1][0]
list2 = alllist[2][0]
prt.append(list0[1:4])
prt.append(list1[1:4])
prt.append(list2[1:4])

y.append(list0[0])

print(list0)
print(list1)
print(list2)
print(prt)
print(y)