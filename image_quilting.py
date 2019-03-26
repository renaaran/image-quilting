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
    def __init__(self, sample, N, ref=False, rot=False):
        self.sample = sample
        self.ref = ref
        self.rot = rot
        self.N = N
        
    def __call__(self, t):
        i = t[0]
        j = t[1]
        t = Pattern(self.sample.take(range(i,i+self.N),mode='raise', axis=0).
                                take(range(j,j+self.N),mode='raise',axis=1), 
                                self.N)
        res = set([t])
        if self.ref:
            res.add(Pattern(np.fliplr(t.data), self.N))
        if self.rot:
            res.add(Pattern(np.rot90(t.data), self.N))
        return os.getpid(), res

# define the possible tiles orientations
class Orientation:
        RIGHT_LEFT = 1
        BOTTOM_TOP = 2
        BOTH = 3
    
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
        p = [np.argmin(self.cost[rows-1,:])]
        for i in range(rows-1, 0, -1):
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
            ov1 = blk1[:,-overlap_size:] # right
            ov2 = blk2[:,:overlap_size] # left
        elif orientation == Orientation.BOTTOM_TOP:
            ov1 = np.transpose(blk1[-overlap_size:,:], (1,0,2)) # bottom
            ov2 = np.transpose(blk2[:overlap_size,:], (1,0,2)) # top            
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
    sl1 = blk1[:,-overlap_size:]
    sl2 = blk2[:,:overlap_size]
    a = path[0]
    if debug:
        join_row = lambda i : np.concatenate((sl1[i,:max(0, a-1)], G, sl2[i,max(a, 1):]))
    else:
        join_row = lambda i : np.concatenate((sl1[i,:a], sl2[i,a:]))
    c = join_row(0)
    res = np.zeros((block_size, overlap_size, 3), dtype=np.int32)
    res[0,:] = c
    for i in range(1, block_size):
        a = path[i]
        c = join_row(i)
        res[i,:] = c
    return np.hstack((res, blk2[:,overlap_size:]))
 
def join_vertical_blocks(blk1, blk2, path, debug=False):
    G = [[0,255,0]] # green pixel
    sl1 = blk1[-overlap_size:,:]
    sl2 = blk2[:overlap_size,:]
    a = path[0]
    if debug:
        join_col = lambda i : np.concatenate((sl1[:max(0, a-1),i], G, sl2[max(a, 1):,i]))
    else:
        join_col = lambda i : np.concatenate((sl1[:a,i], sl2[a:,i]))
    c = join_col(0)
    res = np.zeros((overlap_size, block_size, 3), dtype=np.int32)
    res[:,0] = c
    for i in range(1, block_size):
        a = path[i]
        c = join_col(i)
        res[:,i] = c
    return np.vstack((res, blk2[overlap_size:,:]))
    
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

def get_best(lst, blks, N, orientation):
    pq = []
    pq_N = 1
    heapq.heapify(pq) 
    sample = random.sample(lst, N)
    for patt in sample:
        blk = patt.data
        if orientation != Orientation.BOTH:
            l2 = Minimum_Cost_Path.calc_L2_error(blks[0], blk, block_size, overlap_size, orientation)
            err = l2.sum()
        else:
            l2u = Minimum_Cost_Path.calc_L2_error(blks[0], blk, block_size, overlap_size, Orientation.BOTTOM_TOP)
            l2l = Minimum_Cost_Path.calc_L2_error(blks[1], blk, block_size, overlap_size, Orientation.RIGHT_LEFT)
            err = l2u.sum() + l2l.sum()
        pqe = (-err, blk)
        if len(pq) < pq_N:
            heapq.heappush(pq, pqe)
        else:
            try:
                heapq.heappushpop(pq, pqe)
            except ValueError:
                # skip errors related to duplicate values
                pass            
    return random.sample(pq, 1)[0][1]

def add_block(img, blk, y, x, block_size, overlap_size):
    dx = max(0, x*(block_size-overlap_size))
    dy = max(0, y*(block_size-overlap_size))
    img[dy:dy+block_size,dx:dx+block_size,:] = blk.copy()
    
def get_block(img, y, x, block_size, overlap_size):
    dx = max(0, x*(block_size-overlap_size))
    dy = max(0, y*(block_size-overlap_size))
    return img[dy:dy+block_size,dx:dx+block_size,:].copy()
           
def debug_horizontal_join(blk1, blk2, path, overlap_size, block_size):
    G = [[0,255,0]] # green pixel
    sl1 = blk1[:,-overlap_size:]
    sl2 = blk2[:,:overlap_size]
    bar = np.zeros((block_size, 1, 3), dtype=int)
    img = np.hstack((blk1, bar, sl1, bar, sl2, bar, blk2))  
    plt.imshow(img)
    plt.show()
    a = path[0]
    print(len(path),path)
    concat = lambda i : np.concatenate((sl1[i,:max(0, a-1)], G, sl2[i,max(a, 1):]))
    c = concat(0)
    res = np.zeros((block_size, overlap_size, 3), dtype=np.int32)
    res[0,:] = c
    for i in range(1, block_size):
        a = path[i]
        c = concat(i)
        res[i,:] = c
    img = np.hstack((blk1[:,:-overlap_size], bar, res, bar, blk2[:,overlap_size:]))  
    plt.imshow(img)
    plt.show()
    img = np.hstack((blk1[:,:-overlap_size], res, blk2[:,overlap_size:]))  
    plt.imshow(img)
    plt.show()
   
debug = False
block_size = 30
overlap_size = block_size//6
number_of_tiles_in_output = 10 # output image widht in tiles

IMG_SIZE=(2*(block_size-overlap_size)) + \
    ((number_of_tiles_in_output-2)*(block_size-2*overlap_size)) + \
    ((number_of_tiles_in_output-1)*overlap_size)
    
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    img_source = plt.imread('mesh.jpg')
    img_output = np.zeros(img_source.shape, dtype=np.int32)
    plt.imshow(img_source)
    plt.show()
    
    patterns = PatternsFromSample(img_source, block_size)
    random.shuffle(patterns)
    print('Number of patterns = {}'.format(len(patterns)))
    
    sample_size = len(patterns)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.int)
    for i in range(number_of_tiles_in_output):
        for j in range(number_of_tiles_in_output):
            print('\rProgress : (%d,%d)' % (i,j), end = '', flush=True)
            if i == 0 and j == 0:
                img[:block_size,:block_size] = get_random_pattern(patterns)
                img_dbg = img
            elif i == 0 and j > 0:
                blk1 = get_block(img, 0, j-1, block_size, overlap_size) # up
                blk2 = get_best(patterns, (blk1,), sample_size, Orientation.RIGHT_LEFT)
                mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, Orientation.RIGHT_LEFT) 
                out = join_horizontal_blocks(blk1, blk2, mcp.path, debug=debug)
                add_block(img, out, i, j, block_size, overlap_size)
            elif i > 0 and j == 0:
                blk1 = get_block(img, i-1, 0, block_size, overlap_size) # left
                blk2 = get_best(patterns, (blk1,), sample_size, Orientation.BOTTOM_TOP)
                mcp = Minimum_Cost_Path(blk1, blk2, overlap_size, Orientation.BOTTOM_TOP)
                out = join_vertical_blocks(blk1, blk2, mcp.path, debug=debug)
                add_block(img, out, i, j, block_size, overlap_size)
            elif i > 0 and j > 0:
                blk1 = get_block(img, i-1, j, block_size, overlap_size) # up
                blk2 = get_block(img, i, j-1, block_size, overlap_size) # left
                blk3 = get_best(patterns, (blk1, blk2), sample_size, Orientation.BOTH)
                mcp1 = Minimum_Cost_Path(blk1, blk3, overlap_size, Orientation.BOTTOM_TOP)
                mcp2 = Minimum_Cost_Path(blk2, blk3, overlap_size, Orientation.RIGHT_LEFT)
                out1 = join_vertical_blocks(blk1, blk3, mcp1.path, debug=debug)
                out2 = join_horizontal_blocks(blk2, out1, mcp2.path, debug=debug)
                assert mcp1 != mcp2
                add_block(img, out2, i, j, block_size, overlap_size)                

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()