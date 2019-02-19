'''
This assignment finds the labels using K- means clustering of a given list of data.

To run the program use python kmeans.py <dataset_name> no_of_clusters

Shawn Rahul D'Souza - srd59
'''

import random
import math
import sys

#Euclidian Distance between two d-dimensional points
def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)


    
#K-Means Algorithm
def kmeans(k,datapoints):

    # d - Dimensionality of Datapoints
    d = len(datapoints[0]) 
    
    #Limit our iterations
    Max_Iterations = 1000
    i = 0    
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    
    #Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        cluster_centers += [random.choice(datapoints)]
               
        #Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        #In this particular implementation we want to force K exact clusters.
        #To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        #Update Point's Cluster Alligiance
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            
            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):
                
                dist = eucldist(datapoints[p],cluster_centers[c])
                
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c   # Reassign Point to new Cluster
        
        
        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                else: 
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    
                    
            
            cluster_centers[k] = new_center
    
        
    print ("The clustered labels are:")
    for i in range (len(datapoints)):
        print (cluster[i],'\t', i)
    
    
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

k=int (sys.argv[2])
kmeans(k,data)