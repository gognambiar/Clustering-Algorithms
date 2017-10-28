#!/usr/bin/env python2.7

from operator import itemgetter
import sys
import numpy as np

centroids = {}
current_id = None
current_count = 0
count = 0
centroid_id = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    count += 1
    line = line.strip()

    # parse the input we got from mapper.py
    line = np.array(map(float,line.split()))

    centroid_id = int(line[0])

    point = line[1:]

    if centroid_id not in centroids:
    	centroids[centroid_id] = np.zeros((point.shape[0]))
    centroids[centroid_id] += point

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_id == centroid_id:
        current_count += 1

    else:
        if current_id is not None:
            # write result to STDOUT
            print ' '.join(map(str,centroids[current_id]/current_count))
        current_count = 1
        current_id = centroid_id

# do not forget to output the last word if needed!
if current_id == centroid_id:
    print ' '.join(map(str,centroids[current_id]/current_count))
