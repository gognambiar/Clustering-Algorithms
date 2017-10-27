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
    if not os.path.exists(filePath):
        return None

    data = np.genfromtxt(filePath, delimiter='\t')

    labels = data[:,1]
    data = data[:,2:]

    return (data,labels)

def generateDistanceMatrix(data):
    return euclidean_distances(data,data)

def calcRand(predicted,actual):
                
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
    
    vrand = (m11+m00)/float(m11+m01+m10+m00)#Calculating Rand Index
    vjac = m11/float(m11+m10+m01)#Calculating Jaccard Index
    return [vrand,vjac]

def generateClusters(distMatrix, k):
    dist = copy.copy(distMatrix)
    clusters = [[i] for i in np.arange(dist.shape[0])]
    for iteration in xrange(dist.shape[0] - k):

        minIndex =  np.argwhere(dist == np.min(dist[np.nonzero(dist)]))[0]

        dist = np.delete(dist, minIndex, 0)
        dist = np.delete(dist, minIndex, 1)

        clusters.append(clusters[minIndex[0]] + clusters[minIndex[1]])

        clusters = list(np.delete(clusters, minIndex, 0))

        dist = np.vstack((dist, np.zeros(dist.shape[0])))
        dist = np.hstack((dist, np.zeros(dist.shape[0]).reshape(dist.shape[0],1)))

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
    sklearn_pca = sklearnPCA(n_components=2)
    newData = sklearn_pca.fit_transform(data)
    xval = newData[:,0]
    yval = newData[:,1]
    lbls = set(labels)
    #(my_labels)
    fig1 = plt.figure(1)
    #print(lbls)
    for lbl in lbls:
        #cond = my_labels == lbl
        cond = [i for i, x in enumerate(labels) if x == lbl]
        plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(numpoints=1)
    plt.subplots_adjust(bottom=.20, left=.20)
    fig1.suptitle("PCA plot for centroids in "+inputFile.split("/")[-1],fontsize=20)
    if store:
        fig1.savefig("_".join([outputFile,inputFile.split("/")[-1].split(".")[0]])+".png")
    else:
        plt.show()



def main(argv):
    parser = argparse.ArgumentParser(description='Hierarchial Agglomerative Clustering')
    # optional arguments
    parser.add_argument('-o', '--output', help='Output file to store PCA visualization')
    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    requiredNamed.add_argument('-n', '--num', help='Number of Clusters', required=True, type=int)
    args = parser.parse_args()
    
    inputFile = args.input
    k = args.num
    storePCA = False
    outputFile = None

    if args.output:
        storePCA = True
        outputFile = args.output

    print storePCA, outputFile

    data,labels = loadData(inputFile)

    distMatrix = generateDistanceMatrix(data)

    clusters_map = generateClusters(distMatrix, k)
    clusters_array = np.array([clusters_map[i] for i in xrange(len(clusters_map))])

    randIndex, jaccIndex = calcRand(labels, clusters_map)

    print "Rand Index: ", randIndex
    print "Jaccard Index: ", jaccIndex

    plotPCA(clusters_array, data, inputFile, outputFile, storePCA)


    

if __name__ == "__main__":
    main(sys.argv[1:])

