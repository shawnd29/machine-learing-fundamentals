'''
This assignment calculates Naive Bayes algorithm using the mean and standard deviation of a given list of data.

To run the program use python naive_bayes.py <dataset_name> <test_label_name>

Shawn Rahul D'Souza 
'''


import sys

# Read data from file 

datafile = sys.argv[1]
f= open(datafile,'r')
data=[]
i=0
l=f.readline()
while(l!=''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()
rows=len(data)
cols=len(data[0])
f.close()

#Read labels from file

trainablefile=sys.argv[2]
f=open(trainablefile,'r')
trainlabels={}
n=[]
n.append(0)
n.append(0)
l=f.readline()
while(l!=''):
    a=l.split()
    trainlabels[int(a[1])]=int(a[0])
    l=f.readline()
    n[int(a[0])]+=1

#Find mean of each class
m0=[]
m1=[]
for i in range(0,cols,1):
    m0.append(0.001)
    m1.append(0.001)
for i in range(0,rows,1):
    if (trainlabels.get(i)!= None and trainlabels[i]==0):
        for j in range(0,cols,1):
            m0[j]+=data[i][j]
    if (trainlabels.get(i)!= None and trainlabels[i]==1):
        for j in range(0,cols,1):
            m1[j]+=data[i][j]

for j in range(0,cols,1):
    m0[j]/=n[0]
    m1[j]/=n[1]


#Finding standard deviation 

s0=[]
s1=[]

for i in range(0,cols,1):
    s0.append(0)
    s1.append(0)
for i in range(0,rows,1):
    if (trainlabels.get(i)!= None and trainlabels[i]==0):
        for j in range(0,cols,1):
            s0[j]=s0[j]+(data[i][j]-m0[j])**2
    if (trainlabels.get(i)!= None and trainlabels[i]==1):
        for j in range(0,cols,1):
            s1[j]=s1[j]+(data[i][j]-m1[j])**2

for j in range(0,cols,1):
    s0[j]=(s0[j]/(n[0]-1)) **.5
    s1[j]=(s1[j]/(n[1]-1)) **.5
    

#Finding the likelihood

p0=[]
p1=[]

for i in range(0,cols,1):
    p0.append(0)
    p1.append(0)

print("The prediction values are as follows:")
print ("Class \t Value")
for i in range(0,rows,1):
    if (trainlabels.get(i)==None):
        p0=0
        p1=0
        for j in range (0,cols,1):
            p0=((data[i][j]-m0[j])/s0[j])**2
            p1=((data[i][j]-m1[j])/s0[j])**2
        if (p0<p1):
            print ("0 \t", i)
        else: 
            print ("1 \t", i)

