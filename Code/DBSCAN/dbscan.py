import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import os, sys, copy
import argparse
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt

#global dictionary for cluster number with gene_id
cluster = {}
#global list to store the distance matrix
distMatrix = []
#global matrix to store visited points
visited = []
#predicted label for gene_id based on the clustering algorithm we wrote
predicted_labels = []


#method to get filepath and read data file
#strips the column of gene_id and ground truth and stores ground truth as labels
def loadData(filePath):
    # print(filePath)
    if not os.path.exists(filePath):
        return None

    data = np.genfromtxt(filePath, delimiter='\t')
    labels = data[:,1]
    data = data[:,2:]

    return (data,labels)

#method to generate distance matrix from given dataset
def generateDistanceMatrix(data):
    return euclidean_distances(data,data)

#method to implement dbScan algorithm
def dbScan(data, eps, minPts):
    global visited
    global cluster
    global predicted_labels
    c = 0 #cluster number
    visited = [False] * data.shape[0]
    predicted_labels= [0] * data.shape[0]
    for gene_id in range(0, data.shape[0]):
        #if gene_id is not visited mark it visited
        if not visited[gene_id]:
            visited[gene_id] = True
            #get neighbor points for P
            NeighborPts = regionQuery(data, gene_id, eps)
            #if total neighbor points less than minPts, mark it as noise
            if len(NeighborPts) < minPts:
                cluster[gene_id] = -1
                predicted_labels[gene_id]= -1
            #If point has more than or equal to minPts neighbors
            #expand cluster after inctrmenting cluster value for new cluster
            else:
                c = c+1
                expandCluster(data, gene_id, NeighborPts, c, eps, minPts)
                #c=c+1
          
#method to expand cluster
def expandCluster(data, gene_id, NeighborPts, c, eps, minPts):
    global cluster
    global visited
    global predicted_labels
    #add points P in the cluster C
    cluster[gene_id] = c
    predicted_labels[gene_id] = c
    # for all neighbor points P_dash of P
    while True:
        if len(NeighborPts) == 0:
            break
        P_dash = NeighborPts.pop()
        #if point P_dash is not visited, mark it visted and look for its neighbors
        if not visited[P_dash]:
            visited[P_dash] = True
            NeighborPts_dash = regionQuery(data, P_dash, eps)
            #if number of neighbor points of P_dash is greater than or equal to minPts
            #add all neighbors of P_dash to neighbors of P
            if len(NeighborPts_dash) >= minPts:
                NeighborPts.extend(NeighborPts_dash)
        #check if a point P_dash was marked as noise before lies as a neighbor point for any other point,
        #if so add add that point to cluster C
        if visited[P_dash] and P_dash in cluster and cluster[P_dash] == -1 :
            cluster[P_dash] = c
            predicted_labels[P_dash] = c
        #if P_dash is not labelled, mark it in cluster C
        if not P_dash in cluster:
                cluster[P_dash] = c
                predicted_labels[P_dash]=c
   
#method to check in distance matrix to return a list of neighbor points for any point P
def regionQuery(data, gene_id, eps):
    neighbors = []
    global distMatrix
    for i in range(0, data.shape[0]):
        if (distMatrix[i][gene_id] < eps):
            neighbors.append(i)
    return neighbors

#method to calculate external indicex using ground truth labels and predicted labels
def calcIndexes(clusters, ground_truth):
    predicted = np.zeros((ground_truth.shape[0],ground_truth.shape[0]))
    actual = np.zeros((ground_truth.shape[0], ground_truth.shape[0]))

    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[0]):
            if cluster[i] == cluster[j]:
                predicted[i][j] = 1
            if ground_truth[i] == ground_truth[j]:
                actual[i][j] = 1

    m00,m01,m10,m11 = 0,0,0,0
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[0]):
            if(predicted[i][j] == 1 and actual[i][j] == 1):
                m11 += 1
            elif(predicted[i][j] == 0 and actual[i][j] == 1):
                m01 += 1
            elif(predicted[i][j] == 1 and actual[i][j] == 0):
                m10 += 1
            else:
                m00 += 1
    
    randIndex = (m11+m00)/float(m11+m01+m10+m00)#Calculating Rand Index
    jacIndex = m11/float(m11+m10+m01)#Calculating Jaccard Index
    return [randIndex,jacIndex]

#method to plot PCS graphs for dataset based on predicted labels
def plotPCA(labels, data, inputFile, outputFile, store=False):
    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_pca.fit(data)
    newData = sklearn_pca.transform(data)
    xval = newData[:,0]
    yval = newData[:,1]
    lbls = set(labels)
    #(predicted_labels)
    fig1 = plt.figure(1)
    #print(lbls)
    for lbl in lbls:
        #cond = predicted_labels == lbl
        cond = [i for i, x in enumerate(labels) if x == lbl]
        plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(numpoints=1, loc=0, fontsize = 'x-small')
    plt.subplots_adjust(bottom=.20, left=.20)
    plt.grid()
    fig1.suptitle("PCA plot for DBSCAN in "+inputFile.split("/")[-1],fontsize=20)
    if store:
        fig1.savefig("_".join([outputFile,inputFile.split("/")[-1].split(".")[0]])+".png")
    else:
        plt.show()

def main(argv):
    global distMatrix
    global predicted_labels
    parser = argparse.ArgumentParser(description='DBSCAN Clustering')
    # optional arguments
    parser.add_argument('-o', '--output', help='Output file to store PCA visualization')
    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    requiredNamed.add_argument('-e', '--eps', help='Value of epsilon, The maximum distance between two points for them to be considered as in the same cluster.', required=True, type=float)
    requiredNamed.add_argument('-m', '--minpts', help='Value of minpoints, The number of points in a cluster for a point to be considered as a core point. This includes the point itself', required=True, type=int)
    args = parser.parse_args()
    
    inputFile = args.input
    eps = args.eps
    minPts = args.minpts
    storePCA = False
    outputFile = None

    #if output file is specified, plots will be saved as png file
    if args.output:
        storePCA = True
        outputFile = args.output

    # print storePCA, outputFile

    data,labels = loadData(inputFile)

    distMatrix = generateDistanceMatrix(data)

    dbScan(data, eps, minPts)
      
    randIndex, jaccIndex = calcIndexes(cluster, labels)
    print("Rand Index: ", randIndex)
    print("Jaccard Index: ", jaccIndex)
    print("Number of Clusters: ",len([i for i in set(predicted_labels) if i != -1]))

    plotPCA(predicted_labels , data, inputFile, outputFile, storePCA)

    


if __name__ == "__main__":
    main(sys.argv[1:])