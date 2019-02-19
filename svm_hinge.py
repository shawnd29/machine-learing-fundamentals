'''
This assignment optimizes the SVM hinge loss using the gradient descent algorithm.

To run the program use python svm_hinge.py <dataset_name> <test_label_name>

Shawn Rahul D'Souza
'''

import sys
from random import random

#Find the dot product 
def dot_prod(list1, list2):
    dp = 0
    refw = list1
    refx = list2
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
    if int(a[0]) == 0:
        trainlabels[int(a[1])] = -1
    else:
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
eta = 0.0001
k=0
wt = 0.0

#calculate error outside the loop
error=0.0
for i in range (rows):
    if(trainlabels.get(i) != None):
        wt = (trainlabels.get(i))*dot_prod(w,data[i])
        error += max(0.0, (1.0 - wt))
print(error)
curr_error=0.0

while abs (error - curr_error)> 0.000000001:
    k+=1
    delf = []
    for i in range(cols):
        delf.append(0)

#compute gradient
    for i in range(rows):
        if(trainlabels.get(i) != None):
            d_p = dot_prod(w, data[i])
            wt = (trainlabels.get(i)*dot_prod(w,data[i]))
            for j in range (cols):
                 if ( wt < 1):
                     delf[j] += (-1 * (trainlabels.get(i) * data[i][j]))
                 elif(wt >= 1):
                     delf[j] += 0

#update
    for j in range(cols):
        w[j] = w[j] - eta*delf[j]

#compute error
    curr_error = 0
    for i in range (rows):
        if(trainlabels.get(i) != None):
            wt = (trainlabels.get(i))*dot_prod(w,data[i])
            curr_error += max(0, (1.0 - wt))
    print(curr_error,k)
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

for i in range(rows):
    if(trainlabels.get(i) == None):
        d_p = dot_prod(w, data[i])
        if(d_p > 0):
            print("1",i)
        else:
            print("0",i)


