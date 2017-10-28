import pandas as pd
import numpy as np
from random import randint
import sys
import getopt
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt


#Function to calculate centroid
def calcent(centroid_array,centroid_of_elements,original_data):
	global ctr
	new_centroid_array = [centroid_array[i] for i in range(len(centroid_array))]
	for i in range(original_data.shape[1]):
		minv = 1000
		for k in range(len(centroid_array)):
			if(np.linalg.norm(original_data[i]-centroid_array[k]) < minv):
				minv = np.linalg.norm(original_data[i]-centroid_array[k])#Finding the distance between points and the centroid
				centroid_of_elements[i] = centroid_array[k]
		
	for k in range(len(centroid_array)):
		sm = []
		for i in range(original_data.shape[1]):
			if(np.array_equal(centroid_of_elements[i],centroid_array[k])):
				sm.append(original_data[i])
		new_centroid_array[k] = np.average(np.array(sm),axis=0)#Finding the centroid of the cluster

	ctr += 1
	if((not np.array_equal(centroid_array,new_centroid_array)) and num_of_iterations <= num_of_iterations):#Maximum number of iterations is given as 10
		centroid_array,centroid_of_elements,ctr = calcent(new_centroid_array,centroid_of_elements,original_data)
	
	return centroid_array,centroid_of_elements,ctr

#Function to calculate Rand and Jaccard Index
def calcrand(centroid_array,centroid_of_elements,ground_truth):
	cluster_number_elements = np.zeros(len(centroid_of_elements))
	for k in range(len(centroid_array)):
		for i in range(len(centroid_of_elements)):
			if(np.array_equal(centroid_of_elements[i],centroid_array[k])):
				cluster_number_elements[i] = k+1#Initializing a 1 dimesnional array with all the results
				
	compare_result = np.zeros((len(centroid_of_elements),len(centroid_of_elements)))
	compare_ground_truth = np.zeros((len(centroid_of_elements),len(centroid_of_elements)))
	
	for i in range(len(centroid_of_elements)):
		for j in range(len(centroid_of_elements)):
			if(np.array_equal(centroid_of_elements[i],centroid_of_elements[j])):
				compare_result[i][j] = 1#Initializing a 2D array with the results to calculate Rand and Jaccard Index
				
	for i in range(len(centroid_of_elements)):
		for j in range(len(centroid_of_elements)):
			if(np.array_equal(ground_truth[i],ground_truth[j])):
				compare_ground_truth[i][j] = 1#Initializing a 2D array with the ground truth to calculate Rand and Jaccard Index
				
	m00,m01,m10,m11 = 0,0,0,0
	
	#Comapring the values to find m00,m01,m10,m11
	for i in range(len(centroid_of_elements)):
		for j in range(len(centroid_of_elements)):
			if(compare_result[i][j] == 1 and compare_ground_truth[i][j] == 1):
				m11 += 1
			elif(compare_result[i][j] == 0 and compare_ground_truth[i][j] == 1):
				m01 += 1
			elif(compare_result[i][j] == 1 and compare_ground_truth[i][j] == 0):
				m10 += 1
			else:
				m00 += 1
	
	value_rand = (m11+m00)/float(m11+m01+m10+m00)#Calculating Rand Index
	value_jaccard = m11/float(m11+m10+m01)#Calculating Jaccard Index
	return [value_rand,value_jaccard,cluster_number_elements]

def plotPCA(cluster_number_elements,orig_data_frames,file_name,storePCA,outputFile):
	sklearn_pca = sklearnPCA(n_components=2)
	Y = sklearn_pca.fit_transform(orig_data_frames)#Calling PCA function
	xval = pd.DataFrame(Y)[0]
	yval = pd.DataFrame(Y)[1]
	lbls = set(cluster_number_elements)
	fig1 = plt.figure(1)
	for lbl in lbls:
		cond = cluster_number_elements == lbl
		plt.plot(xval[cond], yval[cond], linestyle='none', marker='o', label=lbl)

	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.legend(numpoints=1, fontsize = 'x-small', loc=0)
	plt.subplots_adjust(bottom=.20, left=.20)
	plt.grid()
	fig1.suptitle("PCA plot for clusters in "+file_name.split("/")[-1],fontsize=20)
	#fig1.savefig("PCA_"+file_name+".png")#Plotting the results based on PCA
	if(storePCA == True):
		fig1.savefig("_".join([outputFile,file_name.split("/")[-1].split(".")[0]])+".png")#Plotting the results based on PCA
	else:
		plt.show()


def main():
	import argparse
	global num_of_iterations
	parser = argparse.ArgumentParser(description='K-means Clustering')
	# optional arguments
	parser.add_argument('-o', '--output', help='Output file to store PCA visualization')
	# required arguments
	requiredNamed = parser.add_argument_group('Required named arguments')
	requiredNamed.add_argument('-i', '--input', help='Input data file name', required=True, type=str)
	requiredNamed.add_argument('-n', '--num', help='Number of Clusters', required=True, type=int)
	requiredNamed.add_argument('-t', '--iteration', help='Number of Iterations', required=True, type=int)
	requiredNamed.add_argument('-a', '--centarray', help='Initial Centroid Gene IDs', required=True, type=str)
	args = parser.parse_args()

	file_name = args.input
	p = args.num
	num_of_iterations = args.iteration
	initial_points = args.centarray
	storePCA = False
	outputFile = None

	if args.output:
	    storePCA = True
	    outputFile = args.output

	initial_points = initial_points.replace("[","").replace("]","")
	initial_points = initial_points.split(",")
	initial_points = [int(i) for i in initial_points]		
	if(len(initial_points) != p):
		print("Please enter the initial centroids for all the clusters")
		sys.exit(2)

	global ctr
	ctr = 1
	data_from_file = pd.read_csv(file_name,sep='\t',header=None)#Reading the file as Pandas dataframe so that we can treat each column as an element
	orig_data_frames = data_from_file.drop([0,1], 1)#Dropping the column number and ground truth columns from data
	ground_truth = data_from_file[1]#Storing ground truth column
	orig_data_frames.columns = [i for i in range(orig_data_frames.shape[1])]#Renaming the columns from 0 to n
	original_data = orig_data_frames.transpose()#Taking transpose so that we can treat the data like normal array where we can access rows
	initial_points = [x-1 for x in initial_points]
	initial_points.sort()
	centroid_array = [original_data[i] for i in initial_points]#Picking centroids for the initial points
	centroid_array = np.array(centroid_array)
	centroid_of_elements = [[] for i in range(orig_data_frames.shape[0])]
	centroid_array,centroid_of_elements,ctr = calcent(centroid_array,centroid_of_elements,original_data)#Calculate Centroid
	value_rand,value_jaccard,cluster_number_elements = calcrand(centroid_array,centroid_of_elements,ground_truth)#Calculate Rand and Jaccard Indexes
	print("Gene ID"+"\t"+"Cluster Number")
	for i in  range(len(cluster_number_elements)):
		print(str(i+1)+"\t"+str(cluster_number_elements[i]))
	print("The Rand Index is " + str(value_rand))
	print("The Jaccard Index is " + str(value_jaccard))
	plotPCA(cluster_number_elements,orig_data_frames,file_name,storePCA,outputFile)#Plot PCA graph

if __name__ == '__main__':
	main()
