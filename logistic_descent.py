'''
This assignment optimizes using the logistic discrimination gradient
descent algorithm.

To run the program use python logistic_descent.py <dataset_name> <test_label_name>

Shawn Rahul D'Souza  
'''

import sys
import math
from random import random

#Find the dot product 
def dot_prod(w, k):
    dp = 0
    refw = w
    refx = k
    for j in range (cols):
        dp += refw[j] * refx[j]
    return dp

#read data file
datafile = sys.argv[1]
f = open (datafile)
data = []
i = 0
l = f.readline()
while(l != '') : 
    a = l.split()
    b = len(a)
    l2 = []
    for j in range(0, b, 1):
        l2.append(float(a[j]))
        if j == (b-1) :
            l2.append(float(1))
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])

f.close()

#read label data
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)

l = f.readline()
while(l != '') :
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1
f.close()

#initialize w
w = []
for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random()) - 0.01

#initialize delf
delf = []
for i in range(cols):
    delf.append(0)

#gradient descent learning rate.
eta = 0.01
k=0
wt = 0.0

#calculate error outside the loop
error=0.0
for i in range (rows):
    if(trainlabels.get(i) != None):
        y=trainlabels.get(i)
        mul=dot_prod(w,data[i])
        wt = y*mul
        sig= (1/(1 + math.exp(-1*mul)))
        error += (-y*math.log(sig))-((1-y)*math.log((1-sig)))
print(error)
curr_error=0.0

while abs (error - curr_error)< 0.000000001:
    k+=1
    delf = []
    for i in range(cols):
        delf.append(0)

#Finidng gradient
    for i in range(rows):
        if(trainlabels.get(i) != None):
            y=trainlabels.get(i)
            mul = dot_prod(w, data[i])
            wt = (trainlabels.get(i)*mul)
            sig = (1/(1 + math.exp(-1*mul)))
            for j in range (cols):
                delf[j] += (sig-y)* data[i][j]          

#update the values
    for j in range(cols):
        w[j] = w[j] - eta*delf[j]

#compute error
    curr_error = 0
    for i in range (rows):
        if(trainlabels.get(i) != None):
            y=trainlabels.get(i)
            mul = dot_prod(w, data[i])
            wt = trainlabels.get(i)*mul
            sig = (1/(1 + math.exp(-1*mul)))
            curr_error+=(-y*math.log(sig))-((1-y)*math.log((1-sig)))
    print(curr_error,k)
    if(abs(error - curr_error)<eta):
        eta = 0.1*eta
    error = curr_error

# calculate differences in error:
print("w =",w)

normw = 0
for j in range((cols-1)):
    normw += w[j]**2
    print(w[j])

normw = (normw)**0.5
print("||w||=", normw)

d_origin = w[(len(w)-1)] / normw
print("d =",d_origin)

#Calculate prediction

print("The prediction values are as follows:")
print ("Class \t Value")
for i in range(rows):
    if(trainlabels.get(i) == None):
        mul = dot_prod(w, data[i])
        if(mul > 0):
            print("1\t",i)
        else:
            print("0\t",i)


