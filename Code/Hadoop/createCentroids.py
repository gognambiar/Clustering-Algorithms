import sys
import numpy as np 
import random

centroid_file = sys.argv[1]
k = int(sys.argv[3])
dataFile = sys.argv[2]

data = np.genfromtxt(dataFile, delimiter='\t')

labels = data[:,1]
data = data[:,2:]

randomIndices = random.sample(range(data.shape[0]), k)

with open(centroid_file, 'w') as writeFile:
	for i in xrange(k):
		writeFile.write(" ".join(map(str,data[randomIndices[i]])) + "\n")
