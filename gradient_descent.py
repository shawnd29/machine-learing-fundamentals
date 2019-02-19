'''
This assignment calculates Gradient Descent algorithm for minimizing the least squares loss of a given list of data.

To run the program use python gradient_descent.py <dataset_name> <test_label_name>

Shawn D'Souza
'''


import sys
from math import sqrt
from random import random


#Dot product function

def dot_product(w,data):
    refw=w
    refx=data
    dp=0
    for j in range (0,cols,1):
        dp+= refx[j]*refw[j]
    return dp

# Read data from file 

datafile = sys.argv[1]
f= open(datafile,'r')

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

#Read labels from file


trainablefile=sys.argv[2]
f=open(trainablefile,'r')
trainlabels={}
n=[]
n.append(0)
n.append(0)
l=f.readline()
while(l != '') : 
    a = l.split()
    if int(a[0]) == 0:
        trainlabels[int(a[1])] = -1
    else:
        trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1



# Initialize w
print ("rows= ",rows)
print("coloumns = ",cols)
w = []

for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random()) - 0.01


#Gradient descent iteration
eta = 0.0001
error = 0

delf = []
for i in range(cols):
    delf.append(0)

flag = 0
k=0
print ("Errors within the labels:")
print("Error\t value")
    
while(flag != 1):
    k+=1
    pre_error = error

    for i in range(rows):
        if(trainlabels.get(i) != None):
            d_p = dot_product(w, data[i])

            #compute gradient

            for j in range (cols):
                delf[j] += (trainlabels.get(i) - d_p) * data[i][j]



#compute error
    error = 0
    for i in range (rows):
        if(trainlabels.get(i) != None):
            error += ( trainlabels.get(i) - dot_product(w,data[i]) )**2
            print(error,'\t',k)
            if(error<0.01):
                eta = 0.000001
                if(error<0.001):
                    eta = 0.0000001
                    if(error<0.00001):
                        eta = 0.00000001
                        if(error<0.000001):
                            flag=1

#update
    for j in range(cols):
        w[j] = w[j] + eta*delf[j]
# print error
normw =0
for j in range (0,cols-1,1):
    normw+=w[j]**2
    print (w[j])
normw=normw**.05
print ("||w|| = ", normw)
d_origin= w[len(w)-1]/normw
print ("distance to origin = ", d_origin)

#Predict labels
print ("\nPrediction:\n")
print("Label\t Values")
for i in range(rows):
    if(trainlabels.get(i) == None):
        d_p = dot_product(w, data[i])
        if(d_p > 0):
            print("1\t",i)
        else:
            print("0\t",i)
