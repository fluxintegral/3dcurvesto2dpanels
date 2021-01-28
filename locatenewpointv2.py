# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:25:03 2020

@author: hendrik.pretorius
"""

import numpy as np
import matplotlib.pylab as plt

#################################################################################
def reduceto2D(points):
    V_store = np.array([[0,0,0]])
    Vector_magnitude = []
    theta_store = []
    for i in range(len(points)-1):
        V = np.array([points[i+1] - points[i]])
        norm = np.linalg.norm(V)
        Vector_magnitude = np.append(Vector_magnitude,norm)
        V_store = np.append(V_store,V,axis=0)
        
    V_store = V_store[1:,:] # Remove first 000 entry used to initialize array
    
    for i in range(len(V_store)-1):
        theta = np.arccos(np.dot(V_store[i,:],V_store[i+1,:])/(np.linalg.norm(V_store[i,:])*np.linalg.norm(V_store[i+1,:])))    
        theta_store = np.append(theta_store,theta)
        
    return(Vector_magnitude,V_store,theta_store) # similar to L2norm,L2,theta1

def planarpointcloud(magnitude,theta):
    ## initialize basevector along x axis
    basepoint_store = np.array([[0,0],[0,magnitude[0]]]) # always start first leg on y axis
    vperp_store = np.array([[1,0]])
    for i in range(0,len(magnitude)-1):
        basevector = basepoint_store[i+1,:]-basepoint_store[i,:]
        
        vperp = np.cross([basevector[0],basevector[1],0],[0,0,1]) # create perpendicular vector
        vperp = vperp[0:2] # third point is always 0 and must be removed 
        
        ##
        '''This function needs further development. As is the code only works if the next point is part of a 
        expanding x,y pattern. This expansion is accomplished by following a theta sign change convention that
        is based on the sequence point 1 theta positive is anti clockwise, point 2 theta positive is clock wise
        etc. This is done by using (-1)**i it should be made more robust.
        '''
        w = basevector*np.cos((-1)**i*theta[i])+vperp*np.sin((-1)**i*theta[i]) # new vector
        ## 
        
        wu = w/np.linalg.norm(w)
        wf = wu*magnitude[i+1]
        
        vperp_store = np.append(vperp_store,np.array([vperp]),axis=0)
        basepoint_store = np.append(basepoint_store,np.array([[basepoint_store[i+1,0]+wf[0],basepoint_store[i+1,1]+wf[1]]]),axis=0)
        
    return(basepoint_store,vperp_store)
    
def conversioncheck(points3d,points2d):
    '''Check that the surface area is maintained in the 3d to 2d conversion '''
    for i in range(len(points3d)-2):
        
        v1 = points3d[i,:] -   points3d[i+1,:]
        v2 = points3d[i+2,:] - points3d[i+1,:]
        c1 = points2d[i,:] -   points2d[i+1,:]
        c2 = points2d[i+2,:] - points2d[i+1,:]
        va = np.linalg.norm(np.cross(v1,v2))
        ca = np.linalg.norm(np.cross(c1,c2))
        if np.round(va - ca,5) != 0:
            print('Error Area Deviance')
        
      
    

#################################################################
pts = np.array([[0,0,0],[0,1,0],[1,.5,0],[1.5,1,0],[2,.8,0],[2.5,1.8,0]])
magnitudes,vz,thetas = reduceto2D(pts)
hope,vpe = planarpointcloud(magnitudes,thetas)
conversioncheck(pts,hope)


plt.plot(hope[:,0],hope[:,1],'--x')
plt.plot(pts[:,0],pts[:,1])



