import sys
from subprocess import Popen,PIPE
import os
import numpy as np

def compareClusters(original, new, tolerance):
	original = np.sort(original, axis=0)
	new = np.sort(new, axis=0)

	difference = np.array([np.linalg.norm(original[i] - new[i]) for i in xrange(original.shape[0])])

	moreThanTol = np.argwhere(difference > tolerance)

	if moreThanTol.shape[0] != 0:
		return False

	return True	

def calcRand(clin,gt):
				
	resrn = np.zeros((len(clin),len(clin)))
	gtrn = np.zeros((len(clin),len(clin)))
	
	for i in range(len(clin)):
		for j in range(len(clin)):
			if(np.array_equal(clin[i],clin[j])):
				resrn[i][j] = 1
				
	for i in range(len(clin)):
		for j in range(len(clin)):
			if(np.array_equal(gt[i],gt[j])):
				gtrn[i][j] = 1
				
	m00,m01,m10,m11 = 0,0,0,0
	
	for i in range(len(clin)):
		for j in range(len(clin)):
			if(resrn[i][j] == 1 and gtrn[i][j] == 1):
				m11 += 1
			elif(resrn[i][j] == 0 and gtrn[i][j] == 1):
				m01 += 1
			elif(resrn[i][j] == 1 and gtrn[i][j] == 0):
				m10 += 1
			else:
				m00 += 1
	
	vrand = (m11+m00)/float(m11+m01+m10+m00)#Calculating Rand Index
	vjac = m11/float(m11+m10+m01)#Calculating Jaccard Index
	return [vrand,vjac]


if len(sys.argv) < 6:
	print 'usage python2.7 launcher.py centroid_file data_file k max_iters tolerance'
	exit(0)

centroid_file = sys.argv[1]
data_file = sys.argv[2]
k = int(sys.argv[3])
max_iters = int(sys.argv[4])
tolerance = float(sys.argv[5])

HADOOP_DIR = os.environ['HADOOP_HOME']
HADOOP_DEL_OUTPUT = "hdfs dfs -rm -r -f outputFile"

HADOOP_GET_CLUSTERS = "hdfs dfs -cat outputFile/*"

HADOOP_RUN_CMD = ['hadoop', 'jar', '/usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar', '--files', 'centroids,mapper.py,reducer.py', '--mapper', '"mapper.py centroids"', '--reducer', 'reducer.py', '--input', 'input/cho.txt', '--output', 'clusters']
HADOOP_RUN_CMD[2] = os.path.join(HADOOP_DIR, 'libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar')
HADOOP_RUN_CMD[4] = ','.join([centroid_file,'mapper.py','reducer.py'])
HADOOP_RUN_CMD[6] = '"mapper.py %s"' % (centroid_file)
HADOOP_RUN_CMD[10] = "inputFile/%s" % (data_file.split("/")[-1])
HADOOP_RUN_CMD[12] = "outputFile"

proc = Popen(("python2.7 createCentroids.py %s %s %s" % (centroid_file, data_file, k)).split(), stdout=PIPE, stderr=PIPE)
proc.wait()
proc = Popen(("hdfs dfs -mkdir -p inputFile").split(), stdout=PIPE, stderr=PIPE)
proc.wait()
proc = Popen(("hdfs dfs -put %s inputFile/." % (data_file)).split(), stdout=PIPE, stderr=PIPE)
proc.wait()
proc = Popen((HADOOP_DEL_OUTPUT).split(), stdout=PIPE, stderr=PIPE)
proc.wait()

centroids = np.genfromtxt(centroid_file)

count = 0
while True:
	count += 1
	print 'Iteration #%s' % count
	if count > max_iters:
		break

	proc = Popen(HADOOP_RUN_CMD, stdout = PIPE, stderr = PIPE)
	std = proc.communicate()
	print std[1]
	if 'completed successfully' in std[1]:
		proc2 = Popen(HADOOP_GET_CLUSTERS.split(), stdout = PIPE, stderr = PIPE)

		newClusters = proc2.communicate()[0]

		newClusters = str(newClusters).replace("\\t"," ").replace("\\n"," ").replace("\t", " ").replace("\n", " ").strip()
		newClusters = np.array(list(map(float, newClusters.split()))).reshape(k,-1)

		if compareClusters(centroids, newClusters, tolerance):
			centroids = newClusters
			break
		else:
			with open(centroid_file,'w') as writer:
				for centroid in newClusters:
					writer.write(" ".join(map(str,centroid)) + "\n")
			centroids = newClusters
		proc = Popen((HADOOP_DEL_OUTPUT).split(), stdout=PIPE, stderr=PIPE)
		proc.wait()


# print centroids


data = np.genfromtxt(data_file)
data = data[:,1:]

CAT_CMD = '''cat %s''' % (data_file)
MAP_CMD = '''python2.7 mapper.py %s''' % (centroid_file)

proc1 = Popen(CAT_CMD.split(), stdout=PIPE)
proc2 = Popen(MAP_CMD.split(), stdin=proc1.stdout, stdout=PIPE, stderr=PIPE)
proc1.wait()
output = proc2.communicate()[0]
new_data = np.fromstring(output, sep="\n").reshape(-1,data.shape[1])

randIndex, jaccIndex = calcRand(data, new_data)

print "Rand Index: ", randIndex
print "Jaccard Index: ", jaccIndex
