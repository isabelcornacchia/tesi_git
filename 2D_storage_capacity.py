#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:30:19 2020

@author: isabel
"""

import numpy as np
#import random
import os



def main():
    
    SimulationName="2D"
    nl=30       #neuroni per lato
    N=nl*nl
    ms=np.linspace(2,20,10)
    gammas=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]     
    samples=5          #iterations for each gamma and m
    L=20.0
    f=0.3
    #fs=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.8]
    
    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)
    
    #OVERLAPS
    print("Starting ...")
    overlaps=np.zeros((len(ms),len(gammas),samples))
    for i in range(len(ms)):
        for j in range(len(gammas)):
            for k in range(samples):
                m=int(ms[i])
                gamma=gammas[j]
                print("Computing retrieval for nmaps="+str(m)+" gamma="+str(gamma)+" sample n: "+str(k+1))
                grid=RegularPfc(N,L,m)
                np.save(SimulationName+"/pfc_"+str(i)+str(j)+str(k),grid)
                J=BuildJ(N,grid,L,gamma)
                Vfinal, th, mean=compute_retrieval(grid,N,L,f,J)
                np.save(SimulationName+"/Vfinal_"+str(i)+str(j)+str(k),Vfinal)
                overlaps=calculate_overlaps(Vfinal,grid,L)
                np.save(SimulationName+"/overlaps_"+str(i)+str(j)+str(k),overlaps)
    print("Dynamics terminated, result saved")
    
    
    #MULTIPLE fs
    '''
    for g in range(len(gammas)):
        for f in range(len(fs)):
            print("Running for g="+str(gammas[g])+" f["+str(f)+"]")
            grid=RegularPfc(N,L,m) # defines environment
            np.save(SimulationName+"/g"+str(gammas[g])+"_pfc_f"+str(fs[f]),grid)
            J=BuildJ(N,grid,L,gammas[g]) # Builds connectivity
            Vfinal, th, mean=compute_retrieval(grid,N,L,fs[f],J)
            np.save(SimulationName+"/g"+str(gammas[g])+"_Vfinal_f"+str(fs[f]),Vfinal)
            np.save(SimulationName+"/g"+str(gammas[g])+"_Threshold_f"+str(fs[f]),th)
            np.save(SimulationName+"/g"+str(gammas[g])+"_mean_f"+str(fs[f]),mean)
    print("Dynamics terminated, result saved")
    #'''
    
    return


# FUNCTIONS
    
def K(x1,x2,L,gamma):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return np.exp(-abs(d))+gamma*np.sign(dx)*np.exp(-abs(dx))
    
def KS(x1,x2,L):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return np.exp(-abs(d))
        
def transfer(h):
        if h>0:
            return h
        else:
            return 0
    
def RegularPfc(N,L,m):
        Nl=int(np.sqrt(N))
        grid=np.zeros((m,N,2))
        tempgrid=np.zeros((N,2))
        for i in range(Nl):
            for j in range(Nl):
                tempgrid[i+Nl*j][0]=i*float(L)/float(Nl)
                tempgrid[i+Nl*j][1]=j*float(L)/float(Nl)
        for j in range(m):
            labels=np.random.permutation(N)
            for k in range(N):
                grid[j][labels[k]]=tempgrid[k]
    
        return grid
    
def BuildJ(N,grid,L,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L,gamma)
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
        maxsteps=100
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec
    
def compute_retrieval(grid,N,L,f,J):
	#V=np.random.uniform(0,1,N)
	V=correlate_activity(grid[0],L)
	V=V/np.mean(V)
	Vfinal, th, mean=dynamics_storage(f,V,N,J)
	return Vfinal, th, mean

def dynamics_storage(f,V,N,J): 
        maxsteps=200
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            th=np.percentile(V,(1.0-f)*100)
            V=Sparsify(V,f)
            mean=np.mean(V)
            V=V/np.mean(V)
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return V, th, mean

def calculate_overlaps(V,grid,L):
	overlaps=np.zeros(len(grid))
	for k in range(len(grid)):
		m=0
		for i in range(len(V)):
			for j in range(i):
				m=m+V[i]*V[j]*KS(grid[k][i],grid[k][j],L)
		m=m/(float(len(V)*(len(V)-1)/2))
		overlaps[k]=m
	return overlaps
	
def correlate_activity(pos,L):
	V=np.zeros(len(pos))
	center=np.array([L/2, L/2])
	for i in range(len(V)):
		V[i]=KS(pos[i],center,L)
	return V

    
if __name__ == "__main__":
    main()