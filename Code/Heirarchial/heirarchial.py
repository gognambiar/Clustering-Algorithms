import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import os
import sys
import getopt
import copy

def loadData(filePath):
	print filePath
	if not os.path.exists(filePath):
		return None

	data = np.genfromtxt(filePath, delimiter='\t')
	# data /= 100
	# print data
	labels = data[:,1]
	data = data[:,2:]

	return (data,labels)

def generateDistanceMatrix(data):
	return euclidean_distances(data,data)

def generateClusters(distMatrix, k):
	dist = copy.copy(distMatrix)
	clusters = [[i] for i in np.arange(dist.shape[0])]
	for iteration in xrange(dist.shape[0] - k):
		print '*'*80
		print dist.shape

		minIndex = 	np.argwhere(dist == np.min(dist[np.nonzero(dist)]))[0]
		print minIndex, clusters[minIndex[0]], clusters[minIndex[1]]

		dist = np.delete(dist, minIndex, 0)
		dist = np.delete(dist, minIndex, 1)

		print dist.shape

		clusters.append(clusters[minIndex[0]] + clusters[minIndex[1]])

		clusters = list(np.delete(clusters, minIndex, 0))

		dist = np.vstack((dist, np.zeros(dist.shape[0])))
		dist = np.hstack((dist, np.zeros(dist.shape[0]).reshape(dist.shape[0],1)))

		print dist.shape
		# print np.array(np.meshgrid(a,b)).T.reshape(len(a)*len(b),-1)
		# print distMatrix

		for i in xrange(len(clusters)-1):
			# print i
			a = clusters[-1]
			b = clusters[i]
			# for j in np.array(np.meshgrid(a,b)).T.reshape(len(a)*len(b),-1):
				# print j, distMatrix[j[0]][j[1]]

			minVal = np.min([distMatrix[k[1]][k[0]] for k in np.array(np.meshgrid(a,b)).T.reshape(-1,2)])
			# minVal = np.min([[distMatrix[j][k] for k in b] for j in a])

			dist[-1][i] = minVal
			dist[i][-1] = minVal

		# print clusters
		# print dist

		print '*'*80

		# break

		# print dist[-1]

		# break

	print clusters

	print dist
		# for 



def main(argv):
    global support
    global confidence
    queryfile = None
    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=", "ofile=", "kvalue="])
    except getopt.GetoptError:
        print('Apriori.py -i <inputfile> -o <outputfile> -k <num clusters>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Apriori.py -i <inputfile> -o <outputfile> -k <num clusters>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-k", "--kvalue"):
            k = int(arg)

    data,labels = loadData(inputfile)
    print data
    # print data[0]
    # print data[1]
    # print np.linalg.norm(data[0]- data[1])
    distMatrix = generateDistanceMatrix(data)

    generateClusters(distMatrix, k)

    # print distMatrix[0][1]
    

if __name__ == "__main__":
    main(sys.argv[1:])

