#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import math
#import random
import os



def main():
    
    SimulationName="D2"
    nl=30   
    N=nl*nl
    m=5
    gammas=np.logspace(-2, math.log10(20), 10)
    L=10.0
    f=0.6
    
    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)
    
    print("Starting dynamics")
    
    #MULTIPLE GAMMAS
    #'''
    for g in range(len(gammas)):
        print("Running for gamma["+str(g)+"]")
        grid=RegularPfc(N,L,m) # defines environment
        np.save(SimulationName+"/pfc_"+str(g),grid)
        J=BuildJ(N,grid,L,gammas[g]) # Builds connectivity
        V=np.random.uniform(0,1,N)
        V=V/np.mean(V)
        Vvec=dynamics(f,V,N,J)
        np.save(SimulationName+"/Vdynamics_"+str(g),Vvec)
    #'''
    
    #SINGLE GAMMA
    '''
    grid=RegularPfc(N,L,m) # defines environment
    np.save(SimulationName+"/pfc_f"+str(f),grid)
    J=BuildJ(N,grid,L,gammas) # Builds connectivity
    V=np.random.uniform(0,1,N)
    V=V/np.mean(V)
    Vvec=dynamics(f,V,N,J)
    np.save(SimulationName+"/Vdynamics_f"+str(f),Vvec)
    '''
    
    print("Dynamics terminated, result saved")
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
        maxsteps=200
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec
    
if __name__ == "__main__":
    main()