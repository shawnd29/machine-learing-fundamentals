'''
This program gives a prediction across Multiple hyperplanes 
and finds the error associated with it.

To run the program use python multi_hyperplanes.py <dataset_name> <test_label_name>

Shawn D'Souza 

'''

import sys
from math import sqrt
from sklearn import svm
import random
from sklearn.model_selection import cross_val_score

def dotProduct(w, x):
    dp = 0.0
    for wi, xi in zip(w, x):
        dp += wi * xi
    return dp
        
def sign(x):
    if(x > 0):
        return 1
    elif(x < 0):
        return -1
    return 0

# Read data

datafile = sys.argv[1]
f=open(datafile)
merged_data=[]  
l=f.readline()   
while (l != ''): 
    a=l.split()
    l2=[]
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    merged_data.append(l2)
    l=f.readline()
    

f.close()

# Read labels

labelfile = sys.argv[2]
f=open(labelfile)
trainlabels= {}  
l=f.readline()   
while(l != ''):
    a=l.split()
    trainlabels[int(a[1])] = int(a[0])
    l=f.readline()
data=[]
testdata=[]
prow = []
for i in range(0,len(merged_data),1):
    if(trainlabels.get(i)==None):
        testdata.append(merged_data[i])
        prow.append(i)
    else:
        data.append(merged_data[i])

noRows=len(data)
noCols =len(data[0])

labels=list(trainlabels.values())
dataSets=data
testDataSets=testdata

#Prediction on the orginal data (without hyperplanes)

print('Original data')
svm_model = svm.SVC(kernel='linear', C=1.0, gamma=1) 
svm_model.fit(dataSets, labels)

p_labels = svm_model.predict(testDataSets)

scores_o = cross_val_score(svm_model, dataSets, labels, cv=5)
scores_o[:]=[1-x for x in scores_o]
print("Calculated error for orginal data: ",scores_o.min())
for i in range(len(p_labels)):
    print(int(p_labels[i]), prow[i])


#Prediction based on different values of k 

planes=[10,100,1000,10000]

for k in planes:
    print('Random hyperplanes data at k = ',k)
    w = []
    for i in range(0, k, 1):
        w.append([])
        for j in range(0, noCols, 1):
            w[i].append(random.uniform(-1, 1))
    z = []
    for i, data in enumerate(dataSets):
        z.append([])
        for j in range(0, k, 1):
            
            z[i].append(sign(dotProduct(w[j], data)))
        
    z1 = []
    for i, data in enumerate(testDataSets):
        z1.append([])
        for j in range(0, k, 1):
            
            z1[i].append(sign(dotProduct(w[j], data)))


    model = svm.SVC(kernel='linear', C=0.001, gamma=1) 
    model.fit(z, labels)

    training_labels = model.predict(z)

    scores = cross_val_score(model, z, labels, cv=5)
    scores[:]= [1-x for x in scores]
    print("Calculated error for new features data at k = ",k,"    ",scores.min())

    p_labels = model.predict(z1)

    print('Predicted labels using Random hyperplanes for k = ',k)

    for i in range(len(p_labels)):
        print(int(p_labels[i]), prow[i])
