#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:57:57 2020

@author: isabel
"""

import numpy as np
import math
#import random
import os



def main():
    
    SimulationName="L1_L2"
    A=100
    Ncost=900
    m=1
    gammas=[0.,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8]
    f=0.05  
    
    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)
    
    print("Starting dynamics")
    
    #MULTIPLE L1
    L1=np.array([10,9,8,7,6,5,4,3])     #asse x
    L2=A/L1                             #asse y
    N1=np.ones((len(L1)))
    N2=np.ones((len(L1)))
    N=np.ones((len(L1)))
    
    for i in range(len(L1)):
        N1[i]=int(round(math.sqrt(Ncost*L1[i]/L2[i])))
        N2[i]=int(Ncost/N1[i])          #meglio int(round())
        N[i]=int(N1[i]*N2[i])
        print("N1,N2,N=", N1[i],N2[i],N[i])
        print("L1/L2=",L1[i]/L2[i])
        for g in range(len(gammas)):
            print("Running for gamma="+str(gammas[g]))
            grid=RegularPfc(int(N1[i]),int(N2[i]),L1[i],L2[i],m)
            np.save(SimulationName+"/ratio"+str(L1[i]/L2[i])+"_pfc_"+str(gammas[g]),grid)
            J=BuildJ(int(N[i]),grid,L1[i],L2[i],gammas[g])
            V=correlate_activity(grid[0],L1[i],L2[i])
            #V=np.random.uniform(0,1,N)
            V=V/np.mean(V)
            Vvec=dynamics(f,V,int(N[i]),J)      #check dynamics output
            np.save(SimulationName+"/ratio"+str(L1[i]/L2[i])+"_Vfinal_"+str(gammas[g]),Vvec)
    
    '''
    #SINGLE L1
    L1=10    
    L2=A/L1
    N1=int(round(math.sqrt(Ncost*L1/L2)))
    N2=int(round(Ncost/N1))
    N=int(N1*N2)
    print("N1,N2,N=", N1,N2,N)
    print("L1/L2,L1,L2=",L1/L2,L1,L2)
    
    for g in range(len(gammas)):
        print("Running for gamma="+str(gammas[g]))
        grid=RegularPfc(N1,N2,L1,L2,m)          #defines environment
        np.save(SimulationName+"/ratio"+str(L1/L2)+"_pfc_"+str(gammas[g]),grid)
        J=BuildJ(N,grid,L1,L2,gammas[g])        #Builds connectivity
        V=correlate_activity(grid[0],L1,L2)
        #V=np.random.uniform(0,1,N)
        V=V/np.mean(V)
        Vvec=dynamics(f,V,N,J)
        np.save(SimulationName+"/ratio"+str(L1/L2)+"_Vfinal_"+str(gammas[g]),Vvec)
    '''    

    
    print("Dynamics terminated, result saved")
    return

# FUNCTIONS
    
def K(x1,x2,L1,L2,gamma):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L1)/2.0:
            dx=dx-L1
        elif dx<-float(L1)/2.0:
            dx=dx+L1
        if dy>float(L2)/2.0:
            dy=dy-L2
        elif dy<-float(L2)/2.0:
            dy=dy+L2
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return np.exp(-abs(d))+gamma*np.sign(dx)*np.exp(-abs(dx))

def KS(x1,x2,L1,L2):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L1)/2.0:
            dx=dx-L1
        elif dx<-float(L1)/2.0:
            dx=dx+L1
        if dy>float(L2)/2.0:
            dy=dy-L2
        elif dy<-float(L2)/2.0:
            dy=dy+L2
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return np.exp(-abs(d))
    
def transfer(h):
        if h>0:
            return h
        else:
            return 0
    
def RegularPfc(N1,N2,L1,L2,m):
        N=int(N1*N2)
        grid=np.zeros((m,N,2))
        tempgrid=np.zeros((N,2))
        for i in range(N1):
            for j in range(N2):
                tempgrid[i+N1*j][0]=i*float(L1)/float(N1)
                tempgrid[i+N1*j][1]=j*float(L2)/float(N2)
        for j in range(m):
            labels=np.random.permutation(N)
            for k in range(N):
                grid[j][labels[k]]=tempgrid[k]
    
        return grid
    
def BuildJ(N,grid,L1,L2,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L1,L2,gamma)
    return J

def Sparsify(V,f):
        vout=V
        th=np.percentile(V,(1.0-f)*100)
        for i in range(len(V)):
            if vout[i]<th:
                vout[i]=0
            else:
                vout[i]=vout[i]-th
        return vout

def dynamics(f,V,N,J): 
        maxsteps=200
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec[maxsteps-1]

def correlate_activity(pos,L1,L2):
	V=np.zeros(len(pos))
	center=np.array([L1/2, L2/2])
	for i in range(len(V)):
		V[i]=KS(pos[i],center,L1,L2)
	return V
    
if __name__ == "__main__":
    main()