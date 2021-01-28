# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:49:13 2020

@author: hendrik.pretorius
"""

import numpy as np

xyz = []
xy1 = np.loadtxt('panel4')

n = len(xy1)

z = np.zeros((n,1))

xyz = np.append(xy1,z,axis = 1)

np.savetxt('panel4.ibl',xyz)

