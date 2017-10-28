#!/usr/bin/env python2.7

import sys
import numpy as np
from pprint import pprint

if len(sys.argv) != 2:
	print "Needs centroid file input"
	exit(0)

centroid_file = sys.argv[1]

centroids = []

# read centroids from file
with open(centroid_file) as readFile:
	for line in readFile.readlines():
		centroids.append(np.array(map(float,line.split())))


def getNearestClusterIndex(point):
	# function to get nearest centroid for a point using euclidean distance

	distances = np.array([np.linalg.norm(point - i) for i in centroids])
	return np.argwhere(distances == np.min(distances))[0]

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # convert the line to data point
    point = np.array(map(float,line.split()))
    point = point[2:]

    # emit the cluster label along with the point
    print "%s\t%s" % (getNearestClusterIndex(point)[0], " ".join(map(str,point)))
    