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
        self.data = data.astype(np.int32)
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
        t = Pattern(self.sample.take(range(i,i+self.N),mode='raise', axis=0).
                                take(range(j,j+self.N),mode='raise',axis=1), 
                                self.N)
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
        # calculate LE error for the overlap region
        self.L2_error = self.calc_L2_error(blk1, blk2, block_size, 
                                              overlap_size, orientation)
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
        self.cost = np.zeros(self.L2_error.shape, dtype=np.int32)
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
            ov1 = np.transpose(blk1[block_size-overlap_size-2:,:], (1,0,2)) # bottom
            ov2 = np.transpose(blk2[:overlap_size+2,:], (1,0,2)) # top            
        assert ov1.shape == ov2.shape
        return ov1, ov2
    
    @staticmethod
    def calc_L2_error(blk1, blk2, block_size, overlap_size, orientation):
        ov1, ov2 = Minimum_Cost_Path.get_overlap(blk1, blk2, block_size, overlap_size, orientation)
        L2_error = np.sum((ov1-ov2)**2, axis=2)
        assert (L2_error >= 0).all() == True
        return L2_error

def join_horizontal_blocks(blk1, blk2, path, debug=False):
    G = [[0,255,0]] # green pixel
    a = mcp.path[0]
    b = blk1.shape[1]
    concat = lambda i : np.concatenate((blk1[i,:b-overlap_size+a], G, blk2[i,a+1:])) \
        if debug else np.concatenate((blk1[i,:b-overlap_size+a], blk2[i,a+1:]))
    c = concat(0)
    res = np.zeros((block_size, len(c), 3), dtype=np.int32)
    res[0,:] = c
    for i in range(1, block_size):
        a = mcp.path[i]-1
        c = concat(i)
        res[i,:] = c
    return res
 
def join_vertical_blocks(blk1, blk2, path, debug=False):
    G = [[0,255,0]] # green pixel
    a = mcp.path[0]
    b = blk1.shape[0]
    concat = lambda i : np.concatenate((blk1[:b-overlap_size+a,i], G, blk2[a+1:,i])) \
        if debug else np.concatenate((blk1[:b-overlap_size+a,i], blk2[a+1:,i]))
    c = concat(0)
    res = np.zeros((len(c), block_size, 3), dtype=np.int32)
    res[:,0] = c
    for i in range(1, block_size):
        a = mcp.path[i]-1
        c = concat(i)
        res[:,i] = c
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
        l2 = Minimum_Cost_Path.calc_L2_error(blk1, blk2, block_size, overlap_size, orientation)
        err = l2.sum()
        pqe = (-err, blk2)
        if len(pq) < pq_N:
            heapq.heappush(pq, pqe)
        else:
            heapq.heappushpop(pq, pqe)
    return random.sample(pq, 1)[0][1]

def join_blocks(blocks):
        img = np.hstack(tuple(blocks))
        return img
            
block_size = 30
overlap_size = block_size//6
number_of_tiles_in_output = 20 # output image widht in tiles

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    img_source = plt.imread('apple.jpg')
    img_output = np.zeros(img_source.shape, dtype=np.int32)
    plt.imshow(img_source)
    plt.show()
    
    patterns = PatternsFromSample(img_source, block_size)
    random.shuffle(patterns)
    print('Number of patterns = {}'.format(len(patterns)))
    
    ########## first row there is only left+right constraint
    img = get_random_pattern(patterns)
    img_dbg = img
    
    for i in range(number_of_tiles_in_output-1):
        blk1 = img[:,img.shape[1]-block_size:]
        blk2 = get_best(patterns, blk1, len(patterns), Orientation.RIGHT_LEFT)
        # calculate the minimum cost path
        mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, Orientation.RIGHT_LEFT)
        # plot blocks side by side
        img = join_horizontal_blocks(img, blk2, mcp.path, debug=False)
        img_dbg = join_horizontal_blocks(img_dbg, blk2, mcp.path, debug=True)

    plt.figure(figsize=(10,10))
    plt.imshow(img_dbg)
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
        
    ########## second row there is left+right and top+bottom constraint
    
    blks = []
    blks_dbg = []
    
    orientation = Orientation.BOTTOM_TOP
    for i in range(img.shape[1]//block_size):
        blk1 = img[:,(i*block_size):(i*block_size)+block_size]        
        blk2 = get_best(patterns, blk1, len(patterns), orientation)
        # calculate the minimum cost path
        mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, orientation)
        # plot blocks side by side
        blks.append(join_vertical_blocks(blk1, blk2, mcp.path, debug=False))
        blks_dbg.append(join_vertical_blocks(blk1, blk2, mcp.path, debug=True))

    
    img_dbg = join_blocks(blks_dbg)
    plt.figure(figsize=(10,10))
    plt.imshow(img_dbg)
    plt.show()
    
    img = join_blocks(blks)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
