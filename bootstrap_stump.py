
'''
This program calculates the descision stump from the column with the
lowest GINI Index and predicts it based on 100 bootstrapped data

To run the program use python bootstrap_stump.py <dataset_name> <test_label_name>

Shawn D'Souza
'''



import sys
import random

##read data file
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


##combine data with labels
def dataprep(datafile, trainlabel):
    datan = []
    for j in range(len(datafile)):
        if(trainlabel.get(j) != None):
            
            datafile[j].append(trainlabel.get(j))
            datan.append(datafile[j])   
    
    return datan

##sample data for bootstraping
def sampdata(datafile):
    datas = []
    w = []
    for j in range(round(0.9*len(datafile))):
        w.append(0)
        w[j] = random.randint(0,round(0.9*len(datafile)))
    for x in w:
        datas.append(datafile[x])
    return datas        

## Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

## Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

## Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups,'gini':b_score}

## Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

## Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

## Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

## Print a decision tree
def print_tree(node, depth=0):
        return {'index':node['index'], 'value':node['value'], 'left':node['left'], 'right':node['right'], 'gini':node['gini']}
   
## Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

datal = dataprep(data,trainlabels)

pred = {}

c = 0
while(c<3):

    #sample data
    datax = sampdata(datal)
    tree = build_tree(datax, 1, 1)
    pt = print_tree(tree)
    tree = build_tree(datax, 1, 1)
    stump = {'index': pt['index'], 'right': pt['right'], 'value': pt['value'], 'left': pt['left'], 'gini': pt['gini']}

    
    # prediction
    pred_list = {}
    for i in range(len(data)):
        if (trainlabels.get(i) == None):
            prediction = predict(stump,data[i])

            pred_list[int(i)] = int(prediction)
           
    for k, v in pred_list.items():
        
        if(c==0):
            pred[k] = int(v)
        else:
            pred[k] += int(v)
    
    c = c + 1
    print (c ,"Rounds of bootstrapped data completed")

print("The prediction values are as follows:")
print ("Class \t Value")
for k, v in pred.items():
    if(v>1):
        v = 1
    else:
        v = 0
    print(v,'\t',k)  
    