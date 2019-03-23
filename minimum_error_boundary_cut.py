#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:35:50 2019

@author: Renato B. Arantes

This is my implementation/interpretation of the article 
"Image Quilting for Textture Synthesis and Transfer"

I'm only interested in minimum error boundary cut.

"""
import os
import sys
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from concurrent import futures
from itertools import product

class PatternsDataset(Dataset):
    def __init__(self, patterns, transform=None):
        self.patterns = patterns
        self.count = len(self.patterns)
        self.transform = transform
        
    def __getitem__(self, index):
        data = self.patterns[index].data.astype(np.float)
        data = self.transform(data)
        return (data, 0)

    def __len__(self):
        return self.count
    
class Pattern:
    def __init__(self, data, N):
        self.data = data.astype(np.int16)
        self.N = N
        
    def __eq__(self, other):
        return (self.data==other.data).all()
    
    def offs(self,d,inv=False):
        d = tuple(map(lambda x:-x, d) if inv else d)
        return Pattern(self.data[max(0,d[0]):min(self.N+d[0],self.N),max(0,d[1]):min(self.N+d[1],self.N)])
    
    def isCompatible(self,d,other,inv=False):
        return (self.offs(d).data==other.offs(d,inv=True).data).all()    

    def __hash__(self):
        return hash(self.data.tobytes())

class CreatePattern:
    def __init__(self, sample, N):
        self.sample = sample
        self.N = N
        
    def __call__(self, t):
        i = t[0]
        j = t[1]
        t = Pattern(self.sample.take(range(i,i+self.N),mode='raise', axis=0).take(range(j,j+self.N),mode='raise',axis=1), self.N)
#        t_ref = Pattern(np.fliplr(t.data), self.N)
#        t_rot = Pattern(np.rot90(t.data), self.N)
        return os.getpid(), set([t])

# define the possible tiles orientations
class Orientation:
        RIGHT_LEFT = 1
        BOTTOM_TOP = 2
    
class Minimum_Cost_Path:
    def __init__(self, blk1, blk2, overlap_size, orientation):
        assert blk1.shape == blk2.shape
        assert blk1.shape[0] == blk2.shape[1]
        # get the overlap regions      
        block_size = blk1.shape[0]
        self.ov1, self.ov2 = self.get_overlap(blk1, blk2, block_size, overlap_size, orientation)
        # calculate LE error for the overlap region
        self.L2_error = self.calc_L2_error(self.ov1, self.ov2)
        # calculate the minimum cost matrix
        self.calc_cost() 
        # now calculate the minimum cost path
        self.path = self.minimum_cost_path()   
        
    def get_cost_at(self, i, j):
        if i < 0 or i >= self.cost.shape[0] or \
           j <= 0 or j >= self.cost.shape[1]-1:
               return sys.maxsize
        return self.cost[i][j]
    
    def get_costs(self, i, j):
        x = self.get_cost_at(i-1, j-1)
        y = self.cost[i-1][j]
        z = self.get_cost_at(i-1, j+1)
        return x, y, z
        
    def min_index(self, i, j):
        x, y, z = self.get_costs(i, j)
        if (x < y): 
            return j-1 if (x < z) else j+1
        else: 
            return j if (y < z) else j+1
                
    def minimum_cost_path(self):
        rows, cols = self.cost.shape
        p = [np.argmin(self.cost[rows-1,1:-1])+1]
        for i in range(rows-1, -1, -1):
            j = p[-1]
            # get the index of smaller cost
            p.append(self.min_index(i, j))
        p.reverse()
        return p
        
    def calc_cost(self):
        self.cost = np.zeros(self.L2_error.shape, dtype=np.int)
        # we don't need to calculate the first row
        self.cost[0,:] = self.L2_error[0,:]
        rows, cols = self.cost.shape        
        for i in range(1, rows):
            for j in range(cols):
                x, y, z = self.get_costs(i, j)
                self.cost[i][j] = min(x, y, z) + self.L2_error[i][j]

    @staticmethod
    def get_overlap(blk1, blk2, block_size, overlap_size, orientation):
        if orientation == Orientation.RIGHT_LEFT:
            ov1 = blk1[:,block_size-overlap_size-2:] # right
            ov2 = blk2[:,:overlap_size+2] # left
        elif orientation == Orientation.BOTTOM_TOP:
            ov1 = blk1[block_size-overlap_size-2:,:] # bottom
            ov2 = blk2[:overlap_size+2,:] # top            
        assert ov1.shape == ov2.shape
        return ov1, ov2
    
    @staticmethod
    def calc_L2_error(ov1, ov2):
        """
        Returns the L2 norm on the overlaped regions
         
        ov1 : overlap region 1
        ov2 : overlap region 2
        """
        L2_error = np.sum((ov1-ov2)**2, axis=2)
        return L2_error

def join_blocks(blk1, blk2, path, debug=False):
    G = [[0,255,0]] # green pixel
    x = mcp.path[0]
    y = blk1.shape[1]
    concat = lambda i : np.concatenate((blk1[i,:y-overlap_size+x], G, blk2[i,x+1:])) \
        if debug else np.concatenate((blk1[i,:y-overlap_size+x], blk2[i,x+1:]))
    c = concat(0)
    res = np.zeros((block_size, len(c), 3), dtype=np.int)
    res[0,:] = c
    for i in range(1, block_size):
        x = mcp.path[i]-1
        c = concat(i)
        res[i,:] = c
    return res
 
def PatternsFromSample(sample, N):
    patts = set()
    h, w, _ = sample.shape
    with futures.ProcessPoolExecutor() as pool:
        createPattern = CreatePattern(sample, N)
        for ident, toappend in pool.map(createPattern, 
                                        product(range(0, h-N), range(0, w-N)),
                                        chunksize=w):
            patts.update(toappend)
    return list(patts)

def get_random_pattern(lst):
    i = random.randint(0, len(lst))
    return lst[i].data

def get_best(lst, blk1, N, orientation):
    pq = []
    pq_N = 1 #np.floor(N*.01)
    heapq.heapify(pq) 
    sample = random.sample(lst, N)
    for patt in sample:
        blk2 = patt.data
        ov1, ov2 = Minimum_Cost_Path.get_overlap(blk1, blk2, block_size, overlap_size, orientation)
        mcp = Minimum_Cost_Path.calc_L2_error(ov1, ov2)
        err = mcp.sum()
        pqe = (-err, blk2)
        if len(pq) < pq_N:
            heapq.heappush(pq, pqe)
        else:
            try:
                heapq.heappushpop(pq, pqe)
            except ValueError as e:
                print(e)
    return random.sample(pq, 1)[0][1]

block_size = 30
overlap_size = block_size//6
number_of_tiles_in_output = 20 # output image widht in tiles

if __name__ == '__main__':
#    random.seed(42)
    img_source = plt.imread('mesh.jpg')
    img_output = np.zeros(img_source.shape, dtype=np.int)
    plt.imshow(img_source)
    plt.show()
    
    patterns = PatternsFromSample(img_source, block_size)
    random.shuffle(patterns)
    print('Number of patterns = {}'.format(len(patterns)))
    
    img = get_random_pattern(patterns)
    img_dbg = img
    
    ########## first row there is only left+right constraint
    orientaion = Orientation.RIGHT_LEFT
    for i in range(number_of_tiles_in_output):
        blk1 = img[:,img.shape[1]-block_size:]
        blk2 = get_best(patterns, blk1, len(patterns), orientaion)
        # calculate the minimum cost path
        mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, orientaion)
        # plot blocks side by side
        img = join_blocks(img, blk2, mcp.path, debug=False)
        img_dbg = join_blocks(img_dbg, blk2, mcp.path, debug=True)
#    ########## second row there is left+right and top+bottom constraint
#    for i in range(img.shape[1]//block_size):
#        blk1 = img[:,(i*block_size):(i*block_size)+block_size]
#        blk2 = get_best(patterns, blk1, len(patterns)//10, Orientation.BOTTOM_TOP)
#        # calculate the minimum cost path
#        mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, Orientation.BOTTOM_TOP)
#        # plot blocks side by side
#        img = join_blocks(img, blk2, mcp.path, debug=False)
#        img_dbg = join_blocks(img_dbg, blk2, mcp.path, debug=True)

    plt.figure(figsize=(10,10))
    plt.imshow(img_dbg)
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
