from subprocess import Popen, PIPE
import numpy as np

startEps = np.arange(0.2,1.5,0.01)
minPts = list(range(2,11))

bestJac = 0

for eps in startEps:
	for pts in minPts:
		print eps,pts
		proc = Popen(("""python2.7 dbscan.py -i ../../Data/cho.txt -e %s -m %s""" % (eps,pts)).split(), stdout=PIPE, stderr=PIPE)
		proc = proc.communicate()
		out =  proc[0]
		# print out
		currentJac = float(out.split("Jaccard Index: ',")[1].split(")")[0])
		numClusters = int(out.split("Clusters: ',")[1].split(")")[0])

		if currentJac > bestJac:
			print 'Got Jaccard %s for eps : %s minPts %s numClusters %s' % (currentJac, eps, pts, numClusters)
			bestJac = currentJac
