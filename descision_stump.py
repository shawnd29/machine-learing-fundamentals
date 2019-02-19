''' 
This program calculates the descision stump from the column with the lowest GINI Index

To run the program use python descision_stump.py <dataset_name> <test_label_name>

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

print (rows,cols)

#Read labels from file

trainablefile=sys.argv[2]
f=open(trainablefile,'r')
trainlabels={}
n=[]
n.append(0)
n.append(0)
l = f.readline()
while(l != '') :
        a = l.split()
        if int(a[0]) == 0:
                trainlabels[int(a[1])] = -1
        else:
                trainlabels[int(a[1])] = int(a[0])
        n[int(a[0])] = n[int(a[0])]+1
        l=f.readline()
f.close()


ginivals = []
split = 0.0
zero = [0, 0]

# Find minimum gini value

for i in range(cols):
        ginivals.append(zero)
gini = 0.0
col = 0

for j in range(cols):
        listcol = [item[j] for item in data]
        key = sorted(range(len(listcol)), key=lambda k: listcol[k])
        listcol.sort()
        ginival = []
        mingini = 0.0
        #split the data according to the -1 label
        for k in range(1,rows,1):
                lsize = k
                rsize = rows - k
                lp = 0
                rp = 0
                for  l in range(k):
                        if (trainlabels.get(key[l]) == -1):
                                lp += 1
                for r in range(k, rows, 1):
                        if (trainlabels.get(key[r]) == -1):
                                rp += 1
                gini = (lp/rows) * (1-lp/lsize) + (rp/rows)* (1-rp/rsize)
                ginival.append(gini)
                mingini = min(ginival)

                if (ginival[k - 1] == float(mingini)):
                        ginivals[j][0] = ginival[k - 1]
                        ginivals[j][1] = k

        if (j == 0):
                gini = ginivals[j][0]

        if (ginivals[j][0] <= gini):
                gini = ginivals[j][0]
                col = j
                split = ginivals[j][1]

                if (split != 0):
                        split = (listcol[split] + listcol[split - 1]) / 2
print("The gini value is:  ", gini, "  For column:  ", col, "  Split at:  ", split)
