import sys
from subprocess import Popen,PIPE
import os
import numpy as np
import argparse
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt

def compareClusters(original, new, tolerance):
    # compares the new centroids with the old centroids to check if it has converged
    # sorts both the arrays and checks if the distance between respective centroids is not more than the tolerance

    original = np.sort(original, axis=0)
    new = np.sort(new, axis=0)

    difference = np.array([np.linalg.norm(original[i] - new[i]) for i in xrange(original.shape[0])])

    # if any difference more than tolerance then return False else return True
    moreThanTol = np.argwhere(difference > tolerance)

    if moreThanTol.shape[0] != 0:
        return False

    return True 

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

def launchHadoop(dataFile, centroidFile, k, max_iters, tolerance, centers, centersDefined):
    # command to remove output folder from hdfs
    HADOOP_DEL_OUTPUT = "hdfs dfs -rm -r -f outputFile"

    # command to get new centroids from hdfs
    HADOOP_GET_CLUSTERS = "hdfs dfs -cat outputFile/*"

    # main hadoop streaming run command
    HADOOP_RUN_CMD = ['hadoop', 'jar', '/usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar', '--files', 'centroids,mapper.py,reducer.py', '--mapper', '"mapper.py centroids"', '--reducer', 'reducer.py', '--input', 'input/cho.txt', '--output', 'clusters', '-numReduceTasks', '2']
    HADOOP_RUN_CMD[2] = hadoopStreaminJarPath
    HADOOP_RUN_CMD[4] = ','.join([centroidFile,'mapper.py','reducer.py'])
    HADOOP_RUN_CMD[6] = '"mapper.py %s"' % (centroidFile)
    HADOOP_RUN_CMD[10] = "inputFile/%s" % (dataFile.split("/")[-1])
    HADOOP_RUN_CMD[12] = "outputFile"
    HADOOP_RUN_CMD[-1] = str(k)

    HADOOP_RUN_CMD = HADOOP_RUN_CMD[:-2]
    # print HADOOP_RUN_CMD

    # command to create initial random centroids or from the gene ids if provided
    createCentroidsExecStr = '''python2.7 createCentroids.py -i %s -n %s -o %s''' % (dataFile, k, centroidFile)

    if centersDefined:
        createCentroidsExecStr += ''' -s %s''' % (",".join(map(str,centers)))


    proc = Popen(createCentroidsExecStr.split(), stdout=PIPE, stderr=PIPE)
    proc.wait()

    # create input files folder and add the files
    proc = Popen(("hdfs dfs -mkdir -p inputFile").split(), stdout=PIPE, stderr=PIPE)
    proc.wait()

    proc = Popen(("hdfs dfs -put %s inputFile/." % (dataFile)).split(), stdout=PIPE, stderr=PIPE)
    proc.wait()

    # remove output directory from any previos runs
    proc = Popen((HADOOP_DEL_OUTPUT).split(), stdout=PIPE, stderr=PIPE)
    proc.wait()

    # get the centroids
    centroids = np.genfromtxt(centroidFile)

    count = 0
    while True:
        # if more than max iters then break
        count += 1
        print 'Iteration #%s' % count
        if count > max_iters:
            break

        # run hadoop
        proc = Popen(HADOOP_RUN_CMD, stdout = PIPE, stderr = PIPE)
        std = proc.communicate()
        # print std

        if 'completed successfully' in std[1]:
            proc2 = Popen(HADOOP_GET_CLUSTERS.split(), stdout = PIPE, stderr = PIPE)

            # get new centroids
            newClusters = proc2.communicate()[0]

            newClusters = str(newClusters).replace("\\t"," ").replace("\\n"," ").replace("\t", " ").replace("\n", " ").strip()
            newClusters = np.array(list(map(float, newClusters.split()))).reshape(k,-1)

            # if converged then break else write the new centroids to centroid file
            if compareClusters(centroids, newClusters, tolerance):
                centroids = newClusters
                break
            else:
                with open(centroidFile,'w') as writer:
                    for centroid in newClusters:
                        writer.write(" ".join(map(str,centroid)) + "\n")
                centroids = newClusters

            # remove output directory from hdfs
            proc = Popen((HADOOP_DEL_OUTPUT).split(), stdout=PIPE, stderr=PIPE)
            proc.wait()

    # get input data
    data = np.genfromtxt(dataFile)
    data = data[:,1:]

    # get new cluster ids for the data
    CAT_CMD = '''cat %s''' % (dataFile)
    MAP_CMD = '''python2.7 mapper.py %s''' % (centroidFile)

    proc1 = Popen(CAT_CMD.split(), stdout=PIPE)
    proc2 = Popen(MAP_CMD.split(), stdin=proc1.stdout, stdout=PIPE, stderr=PIPE)
    proc1.wait()
    output = proc2.communicate()[0]
    new_data = np.fromstring(output, sep="\n").reshape(-1,data.shape[1])

    return new_data, data

def plotPCA( labels, data, inputFile, outputFile, store=False):
    # apply PCA
    sklearn_pca = sklearnPCA(n_components=2)
    newData = sklearn_pca.fit_transform(data)

    # get x and y values
    xval = newData[:,0]
    yval = newData[:,1]
    lbls = set(labels)

    fig1 = plt.figure(1)

    # plot for each label
    for lbl in lbls:
        cond = [i for i, x in enumerate(labels) if x == lbl]
        plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(numpoints=1, loc=0)
    plt.subplots_adjust(bottom=.20, left=.20)
    fig1.suptitle("PCA plot for centroids in "+inputFile.split("/")[-1],fontsize=20)

    # if PCA output parameter given then store the plot else display it
    if store:
        fig1.savefig("_".join([outputFile,inputFile.split("/")[-1].split(".")[0]])+".png")
    else:
        plt.show()

if __name__ == "__main__":

    # add arguments
    parser = argparse.ArgumentParser(description='Hadoop KMeans Launcher File')

    # optional arguments
    parser.add_argument('-s','--centers', help='Comma Separated Gene Ids for centroids [eg. 1,5,2,3]\n Number of centers should be equal to k', type=str)
    parser.add_argument('-m', '--maxIters', help='Max number of Iterations', type=int, default=100)
    parser.add_argument('-t', '--tolerance', help='Tolerance value for cluster comparision', type=float, default=0.00001)
    parser.add_argument('-p', '--pca', help='Output file to store PCA visualization')

    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-d', '--data', help='Data file name', required=True, type=str)
    requiredNamed.add_argument('-n', '--num', help='Number of Clusters', required=True, type=int)
    requiredNamed.add_argument('-c', '--centroid', help='Centroid File name', required=True, type=str)
    requiredNamed.add_argument('-x', '--hadoop', help='Hadoop Streaming Jar Path', required=True, type=str)
    args = parser.parse_args()

    # parse all the arguments
    dataFile = args.data
    centroidFile = args.centroid
    hadoopStreaminJarPath = args.hadoop
    k = args.num
    max_iters = args.maxIters
    tolerance = args.tolerance

    centersDefined = False
    centers = None

    # if centroid ids given then take them 
    if args.centers:
        centers = args.centers
        if len(centers.split(",")) != k:
            print 'Number of centers provided not equal to k'
            exit(0)
        centersDefined = True
        centers = list(map(int,centers.split(",")))

    storePCA = False
    outputPCAFile = None

    # if pca output given then take it
    if args.pca:
        storePCA = True
        outputPCAFile = args.pca

    # run hadoop K Means
    new_data, data = launchHadoop(dataFile, centroidFile, k, max_iters, tolerance, centers, centersDefined)

    # calculate jaccard and rand indexes
    randIndex, jaccIndex = calcRand(new_data[:,0], data[:,0])

    print "Rand Index: ", randIndex
    print "Jaccard Index: ", jaccIndex

    labels = new_data[:,0] + 1
    data = new_data[:,1:]

    # create PCA plots
    plotPCA(labels, data, dataFile, outputPCAFile, storePCA)

    # print ids with cluster labels
    print 'Gene Id\tCluster Id'
    for i in xrange(len(labels)):
        print '%s\t%s' % (i+1,labels[i])
