# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:25:03 2020

@author: hendrik.pretorius
"""

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import numpy as np


#################################################################
ux = np.array([1,0,0])
pts = np.array([[0,0,0],[1,1,.0],[1.5,.5,.0],[2,1,.0],[2.5,3,.0]])
L1 = pts[1,:] - pts[0,:]
L1norm = np.linalg.norm(L1)
theta0 = np.arccos(np.dot(ux,L1)/(np.linalg.norm(ux)*np.linalg.norm(L1)))
#print(np.rad2deg(theta0))
#print(L1norm)

##
L2 = pts[2,:] - pts[1,:] 
L2norm = np.linalg.norm(L2)
theta1 = np.arccos(np.dot(L2,L1)/(np.linalg.norm(L2)*np.linalg.norm(L1)))
#print(np.rad2deg(theta1))
#print(L2norm)
##
L3 = pts[3,:] - pts[2,:] 
L3norm = np.linalg.norm(L3)
theta2 = np.arccos(np.dot(L3,L2)/(np.linalg.norm(L3)*np.linalg.norm(L2)))
#print(np.rad2deg(theta2))
#print(L3norm)
##
L4 = pts[4,:] - pts[3,:] 
L4norm = np.linalg.norm(L4)
theta3 = np.arccos(np.dot(L4,L3)/(np.linalg.norm(L4)*np.linalg.norm(L3)))
#print(np.rad2deg(theta3))
#print(L4norm)

## Calc rotated 2D vector
# pt1
basevector = np.array([1,0])
ej = np.array([0.8632,.4645,]) # random as long as not co linear with u
vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # per to basevector
test = np.dot(vperp,basevector)
#print(test)  # must be zero then vperp is perp to L12
thetainit = np.deg2rad(90)
w = basevector*np.cos(thetainit)+vperp*np.sin(thetainit) # new vector
wu = w/np.linalg.norm(w)
wf = wu*L1norm 

# pt2
basevector = wf
ej = np.array([0.8632,.4645,]) # random as long as not co linear with u
vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # per to basevector

w2 = basevector*np.cos(theta1)+vperp*np.sin(theta1) # new vector
wu2 = w2/np.linalg.norm(w2)
wf2 = wu2*L2norm#np.abs(wu2*L2norm)

# pt3
basevector = wf2
ej = np.array([0.8632,.4645]) # random as long as not co linear with u
vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # per to basevector

w3 = basevector*np.cos(theta2)+vperp*np.sin(theta2) # new vector
wu3 = w3/np.linalg.norm(w3)
wf3 = wu3*L3norm

# pt4
basevector = wf3
ej = np.array([0.8632,.4645]) # random as long as not co linear with u
vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # per to basevector

w4 = basevector*np.cos(theta3)+vperp*np.sin(theta3) # new vector
wu4 = w4/np.linalg.norm(w4)
wf4 = wu4*L4norm

## Area test
v01 = pts[0,:] - pts[1,:] 
v21 = pts[2,:] - pts[1,:] 
v12 = pts[1,:] - pts[2,:] 
v32 = pts[3,:] - pts[2,:] 
v23 = pts[2,:] - pts[3,:] 
v43 = pts[4,:] - pts[3,:]  

a1 = np.linalg.norm(np.cross(v01,v21))
a2 = np.linalg.norm(np.cross(v12,v32))
a3 = np.linalg.norm(np.cross(v23,v43))

print(a1,a2,a3)
    
    
#
#print(np.rad2deg(theta))


#
#
#
#
#plt.figure()
#plt.plot(pts[:,0],pts[:,1])
#plt.plot([0,vperp[0]],[0,vperp[1]],'--',label = 'normal')
#plt.plot(wf[0],wf[1],'o')
#plt.plot(wf2[0]+wf[0],wf2[1]+wf[1],'o')
#plt.plot(wf[0]+wf2[0]+wf3[0],wf[1]+wf2[1]+wf3[1],'o')
#plt.plot(wf[0]+wf2[0]+wf3[0]+wf4[0],wf[1]+wf2[1]+wf3[1]++wf4[1],'o')



#plt.plot([pts[1,0],wf[0]],[pts[1,1],wf[1]],'-x')
#plt.plot(pts[:,0],pts[:,1])
#plt.axis('equal')
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
        

magnitudes,vz,thetas = reduceto2D(pts)


def planarpointcloud(magnitude,theta):
    ## initialize basevector along x axis
    basepoint_store = np.array([[0,0],[0,magnitude[0]]]) # always start first leg on y axis

    for i in range(0,len(magnitude)-1):
        basevector = basepoint_store[i+1,:]-basepoint_store[i,:]
        
    #basevector is the prior vector, base point is the xy coordinate of from which to project
        ej = np.array([0.8632,.4645]) # random choice may not be colinear with basevector
        vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # perpendicular to basevector
        test = np.dot(vperp,basevector) # Test that vperp is perpendicular to basevector
        if np.round(test,5) != 0:
            ej = np.array([0.8632,.4645*.123])
            vperp = ej - np.dot(basevector,ej)*(basevector/np.dot(basevector,basevector)) # perpendicular to basevector
            test = np.dot(vperp,basevector) # Test that vperp is perpendicular to basevector
            if np.round(test,5) != 0:
                print('ERROR!! Random vector colinear')
            
        w = basevector*np.cos(theta[i])+vperp*np.sin(theta[i]) # new vector
        wu = w/np.linalg.norm(w)
        wf = wu*magnitude[i+1]
        basepoint_store = np.append(basepoint_store,np.array([[basepoint_store[i+1,0]+wf[0],basepoint_store[i+1,1]+wf[1]]]),axis=0)
        
    return(basepoint_store)


hope = planarpointcloud(magnitudes,thetas)
plt.plot(hope[:,0],hope[:,1],'x')
plt.plot(pts[:,0],pts[:,1])


h01 = hope[0,:]-hope[1,:]
h21 = hope[2,:]-hope[1,:]
h12 = hope[1,:]-hope[2,:]
h32 = hope[3,:]-hope[2,:]
h23 = hope[2,:]-hope[3,:]
h43 = hope[4,:]-hope[3,:]

ah1 = np.linalg.norm(np.cross(h01,h21))
ah2 = np.linalg.norm(np.cross(h12,h32))
ah3 = np.linalg.norm(np.cross(h23,h43))

print(ah1,ah2,ah3)
