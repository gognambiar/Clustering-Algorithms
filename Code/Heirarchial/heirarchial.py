import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import os
import sys
import getopt
import copy
import argparse
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt



def loadData(filePath):
    # function to load data from file

    # if file doesnt exist then return None
    if not os.path.exists(filePath):
        return None

    # load data from the file
    data = np.genfromtxt(filePath, delimiter='\t')

    # extract labels and data
    labels = data[:,1]
    data = data[:,2:]

    return (data,labels)

def generateDistanceMatrix(data):
    # function to compute euclidean distance matrix for data
    # matrix[i][j] is the euclidean distance between point i and j
    return euclidean_distances(data,data)

def calcRand(predicted,actual):
    # function to calculate jaccard and rand index between the actual and predicated labels

    # create predictedMatrix and actualMatrix
    predictedMatrix = np.zeros((predicted.shape[0],predicted.shape[0]))
    actualMatrix = np.zeros((predicted.shape[0],predicted.shape[0]))
    

    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[0]):
            if(np.array_equal(predicted[i],predicted[j])):
                predictedMatrix[i][j] = 1
                
            if(np.array_equal(actual[i],actual[j])):
                actualMatrix[i][j] = 1
                
    m00,m01,m10,m11 = 0,0,0,0
    
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[0]):
            if(predictedMatrix[i][j] == 1 and actualMatrix[i][j] == 1):
                m11 += 1
            elif(predictedMatrix[i][j] == 0 and actualMatrix[i][j] == 1):
                m01 += 1
            elif(predictedMatrix[i][j] == 1 and actualMatrix[i][j] == 0):
                m10 += 1
            else:
                m00 += 1

    #Calculating Rand Index
    vrand = (m11+m00)/float(m11+m01+m10+m00)

    #Calculating Jaccard Index
    vjac = m11/float(m11+m10+m01)

    return [vrand,vjac]

def generateClusters(distMatrix, k):

    # copy euclidean distance matrix 
    dist = copy.copy(distMatrix)

    # create array of clusters for initial data points
    clusters = [[i] for i in np.arange(dist.shape[0])]

    for iteration in xrange(dist.shape[0] - k):

        # find smallest non zero distance 
        minIndex =  np.argwhere(dist == np.min(dist[np.nonzero(dist)]))[0]

        # remove the rows and columns for the corresponding points
        dist = np.delete(dist, minIndex, 0)
        dist = np.delete(dist, minIndex, 1)

        # create a cluster containing points from the points for smallest distance
        clusters.append(clusters[minIndex[0]] + clusters[minIndex[1]])

        # remove their old clusters
        clusters = list(np.delete(clusters, minIndex, 0))

        # add new row and column for the new cluster
        dist = np.vstack((dist, np.zeros(dist.shape[0])))
        dist = np.hstack((dist, np.zeros(dist.shape[0]).reshape(dist.shape[0],1)))

        # for every other cluster point calculate smallest euclidean distance  
        for i in xrange(len(clusters)-1):
            a = clusters[-1]
            b = clusters[i]

            minVal = np.min([distMatrix[k[1]][k[0]] for k in np.array(np.meshgrid(a,b)).T.reshape(-1,2)])

            dist[-1][i] = minVal
            dist[i][-1] = minVal

    cluster_id = 0
    clusters_map  = {}
    for elems in clusters:
        for point in elems:
            clusters_map[point] = cluster_id
        cluster_id += 1

    return clusters_map

def plotPCA( labels, data, inputFile, outputFile, store=False):
    # apply PCA
    sklearn_pca = sklearnPCA(n_components=2)
    newData = sklearn_pca.fit_transform(data)

    # get x and y values for plot 
    xval = newData[:,0]
    yval = newData[:,1]

    # get labels
    lbls = set(labels)
    fig1 = plt.figure(1)

    # plot each label
    for lbl in lbls:
        cond = [i for i, x in enumerate(labels) if x == lbl]
        plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(numpoints=1,loc=0)
    plt.subplots_adjust(bottom=.20, left=.20)
    fig1.suptitle("PCA plot for clusters in "+inputFile.split("/")[-1],fontsize=20)

    # if store paramter provided then store the plot else display it 
    if store:
        fig1.savefig("_".join([outputFile,inputFile.split("/")[-1].split(".")[0]])+".png")
    else:
        plt.show()



def main(argv):
    # arguments
    parser = argparse.ArgumentParser(description='Hierarchial Agglomerative Clustering')

    # optional arguments
    parser.add_argument('-o', '--output', help='Output file to store PCA visualization')

    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    requiredNamed.add_argument('-n', '--num', help='Number of Clusters', required=True, type=int)
    args = parser.parse_args()
    
    # parse arguments
    inputFile = args.input
    k = args.num
    storePCA = False
    outputFile = None

    # if pca output provided then take it 
    if args.output:
        storePCA = True
        outputFile = args.output

    # load initial data
    data,labels = loadData(inputFile)

    # generate euclidean distance matrix 
    distMatrix = generateDistanceMatrix(data)

    # perform HAC 
    clusters_map = generateClusters(distMatrix, k)

    # change cluster labels to begin from 1 instead of 0
    clusters_array = np.array([clusters_map[i] for i in xrange(len(clusters_map))]) + 1

    # calculate jaccard and rand index
    randIndex, jaccIndex = calcRand(labels, clusters_map)

    print "Rand Index: ", randIndex
    print "Jaccard Index: ", jaccIndex

    # plot pca
    plotPCA(clusters_array, data, inputFile, outputFile, storePCA)

    # print ids with cluster labels
    print 'Gene Id\tCluster Id'
    for i in xrange(len(clusters_array)):
        print '%s\t%s' % (i+1,clusters_array[i])


    

if __name__ == "__main__":
    main(sys.argv[1:])

