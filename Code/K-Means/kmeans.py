import pandas as pd
import numpy as np
from random import randint

#Function to calculate centroid
def calcent(fsam,clin,ttv):
    global ctr
    nsam = [fsam[i] for i in range(len(fsam))]
    for i in range(ttv.shape[1]):
        minv = 1000
        for k in range(len(fsam)):
            if(np.linalg.norm(ttv[i]-fsam[k]) < minv):
                minv = np.linalg.norm(ttv[i]-fsam[k])#Finding the distance between points and the centroid
                clin[i] = fsam[k]
        
    for k in range(len(fsam)):
        sm = []
        for i in range(ttv.shape[1]):
            if(np.array_equal(clin[i],fsam[k])):
                sm.append(ttv[i])
        nsam[k] = np.average(np.array(sm),axis=0)#Finding the centroid of the cluster

    ctr += 1
    if((not np.array_equal(fsam,nsam)) and ctr < 500):#Maximum number of iterations is given as 500
        fsam,clin,ctr = calcent(nsam,clin,ttv)
    
    return fsam,clin,ctr

#Function to calculate Rand and Jaccard Index
def calcrand(fsam,clin,gt):
    c = np.zeros(len(clin))
    ints = 0
    for k in range(len(fsam)):
        for i in range(len(clin)):
            if(np.array_equal(clin[i],fsam[k])):
                c[i] = k+1
                
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

def main():
    global ctr
    ctr = 0
    pddata = pd.read_csv("../../Data/iyer.txt",sep='\t',header=None)
    print pddata.shape
    testv = pddata.drop([0,1], 1)
    gt = pddata[1]
    testv.columns = [i for i in range(testv.shape[1])]
    ttv = testv.transpose()
    rsam=[randint(0,testv.shape[0]-1) for p in range(0,5)]
    #rsam = [5,25,32,100,132]
    rsam.sort()
    fsam = [ttv[i] for i in rsam]
    fsam = np.array(fsam)
    clin = [[] for i in range(testv.shape[0])]
    fsam,clin,ctr = calcent(fsam,clin,ttv)
    vrand,vjac = calcrand(fsam,clin,gt)
    print vrand
    print("The Number of Iterations before converging is " + str(ctr))
    print("The Rand Index is " + str(vrand))
    print("The Jaccard Index is " + str(vjac))

if __name__ == '__main__':
    main()