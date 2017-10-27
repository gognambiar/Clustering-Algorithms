import sys
import numpy as np 
import random
import argparse

def createCentroidFile(data, k, centroidFile, centers, centersDefined):
    if centersDefined:
        with open(centroidFile, 'w') as writeFile:
            for center in centers:
                row = data[data[:,0] == center][:,2:]
                if row.shape[0] == 0:
                    print 'gene id %s not found in data \nPlease provide valid ids' % center
                    exit(-1)
                writeFile.write(" ".join(map(str,row.flatten())) + "\n")

    else:
        randomIndices = random.sample(range(data.shape[0]), k)

        with open(centroidFile, 'w') as writeFile:
            for i in xrange(k):
                writeFile.write(" ".join(map(str,data[randomIndices[i]][2:])) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Centroids File for KMeans')
    parser.add_argument('-s','--center', help='Comma Separated Gene Ids for centroids [eg. 1,5,2,3]\n Number of centers should be equal to k', type=str)
    # required arguments
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
    requiredNamed.add_argument('-n', '--num', help='Number of Clusters', required=True, type=int)
    requiredNamed.add_argument('-o', '--output', help='Output File to store centroids', required=True, type=str)
    args = parser.parse_args()

    inputFile = args.input
    k = args.num
    centroidFile = args.output
    centersDefined = False
    centers = None

    if args.center:
        centers = args.center
        if len(centers.split(",")) != k:
            print 'Number of centers provided not equal to k'
            exit(0)
        centersDefined = True
        centers = list(map(int,centers.split(",")))


    data = np.genfromtxt(inputFile, delimiter='\t')

    createCentroidFile(data, k, centroidFile, centers, centersDefined)

    # labels = data[:,1]
    # data = data[:,2:]


    
