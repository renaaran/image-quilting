#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:35:50 2019

@author: Renato B. Arantes

This is my implementation/interpretation of the article 
"Image Quilting for Textture Synthesis and Transfer"

I'm only interested in minimum error boundary cut.

"""
import sys
import numpy as np
import matplotlib.pyplot as plt

block_size = 30
overlap_size = block_size//4

class Minimum_Cost_Path:
    def __init__(self, ov1, ov2):
        # calculate LE error for the overlap region
        self.L2_error = self.calc_L2_error(ov1, ov2)
        # calculate the minimum cost matrix
        self.cost = self.calc_cost() 
        # now calculate the minimum cost path
        self.path = self.get_minimum_cost_path()   
            
    def get_minimum_cost_path(self):
        rows = self.L2_error.shape[0]
        cols = self.L2_error.shape[1]  
        p = [np.argmin(self.cost[rows-1,1:-1])+1]
        for i in range(rows-1, 0, -1):
            j = p[-1]
            # get the index of smaller cost
            x = self.cost[i-1][j-1] if j > 1 else sys.maxsize
            y = self.cost[i-1][j]
            z = self.cost[i-1][j+1] if j < cols-1 else sys.maxsize
            if (x < y): 
                p.append(j-1 if (x < z) else j+1) 
            else: 
                p.append(j if (y < z) else j+1)          
        p.reverse()
        return p
        
    def calc_cost(self):
        cost = np.zeros(self.L2_error.shape, dtype=np.uint)
        rows = self.L2_error.shape[0]
        cols = self.L2_error.shape[1]        
        cost[0,:] = self.L2_error[0,:]
        for i in range(1, rows):
            for j in range(cols):
                e = min(cost[i-1][j-1] if j > 0 else sys.maxsize, 
                        cost[i-1][j], 
                        cost[i-1][j+1] if j < cols-1 else sys.maxsize)
                cost[i][j] = e + self.L2_error[i][j]
        return cost
    
    def calc_L2_error(self, ov1, ov2):
        """
        Returns the L2 norm on the overlaped regions
         
        ov1 : overlap region 1
        ov2 : overlap region 2
        """
        L2_error = np.sum((ov1-ov2)**2, axis=2)
        return L2_error

def join_debug(block1, block2, path):
    x = mcp.path[0]
    y = block1.shape[1]
    c = np.concatenate((block1[0,:y-overlap_size+x-1], [[0,255,0]], block2[0,x+1:]))
    res = np.zeros((block_size, len(c), 3), dtype=np.uint8)
    res[0,:] = c
    for i in range(1, block_size):
        x = mcp.path[i]-1
        c = np.concatenate((block1[i,:y-overlap_size+x-1], [[0,255,0]], block2[i,x+1:]))
        res[i,:] = c
    return res

def join(block1, block2, path):
    x = mcp.path[0]
    y = block1.shape[1]
    c = np.concatenate((block1[0,:y-overlap_size+x-1], block2[0,x+1:]))
    res = np.zeros((block_size, len(c), 3), dtype=np.uint8)
    res[0,:] = c
    for i in range(1, block_size):
        x = mcp.path[i]-1
        c = np.concatenate((block1[i,:y-overlap_size+x-1], block2[i,x+1:]))
        res[i,:] = c
    return res
 
img_source = plt.imread('mesh.jpg')
img_output = np.zeros(img_source.shape, dtype=np.uint8)
plt.imshow(img_source)
plt.show()

block1 = img_source[:block_size,:block_size]
block2 = img_source[10:block_size+10,3:block_size+3]
block3 = img_source[23:block_size+23,13:block_size+13]
# plot blocks side by side
ovl = np.hstack((block1, block2))
plt.figure(figsize=(3,2))
plt.imshow(ovl)
plt.show()
# get the overlap regions
ov1 = block1[:,block_size-overlap_size-2:] # right
ov2 = block2[:,:overlap_size+2] # left
# plt overlap regions
ovl = np.hstack((ov1, ov2))
plt.figure(figsize=(3,2))
plt.imshow(ovl)
plt.show()
# calculate the minimum cost path
mcp = Minimum_Cost_Path(ov1, ov2)
# plot blocks side by side
img = join_debug(block1, block2, mcp.path)
plt.figure(figsize=(2,3))
plt.imshow(img)
plt.show()

###################### BLOCK 2+3

ov2 = block2[:,block_size-overlap_size-2:] # right
ov3 = block3[:,:overlap_size+2] # left

# plot blocks side by side
ovl = np.hstack((img, block3))
plt.figure(figsize=(3,2))
plt.imshow(ovl)
plt.show()
# plt overlap regions
ovl = np.hstack((ov2, ov3))
plt.figure(figsize=(3,2))
plt.imshow(ovl)
plt.show()
# calculate the minimum cost path
mcp = Minimum_Cost_Path(ov2, ov3)
# plot blocks side by side
img = join_debug(img, block3, mcp.path)
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.show()

