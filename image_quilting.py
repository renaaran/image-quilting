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
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from concurrent import futures
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--debug', required=False, default=True, help='debug mode')
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')
parser.add_argument('--inputImage', required=True, help='to be processed image path')
parser.add_argument('--outputFolder', required=True, help='folder to output images')

opt = parser.parse_args()

basename = os.path.basename(opt.inputImage)
stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
opt.outputFolder += '/' + basename + '/image_quilting/' + stamp + "/"
if not os.access(opt.outputFolder, os.F_OK):
    os.makedirs(opt.outputFolder)
else:
    raise Exception('Output folder already exists: {}'.format(opt.outputFolder))

print ("outputFolder:"+opt.outputFolder)

text_file = open(opt.outputFolder+"options.txt", "w")
text_file.write(str(opt))
text_file.close()
print (opt)

class PatternsDataset():
    def __init__(self, patterns, transform=None):
        self.patterns = patterns
        self.count = len(self.patterns)
        self.transform = transform

    def __getitem__(self, index):
        data = self.patterns[index].data.astype(np.float32)
        data = self.transform(data)
        return (data, 0)

    def __len__(self):
        return self.countiq.patterns[0].data

class Pattern:
    def __init__(self, data, N):
        self.data = data.astype(np.float32)
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
        if self.ref: res.add(Pattern(np.fliplr(t.data), self.N))
        if self.rot: res.add(Pattern(np.rot90(t.data), self.N))
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
        self.cost = np.zeros(self.L2_error.shape, dtype=np.float32)
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
            ov1 = blk1[:,-overlap_size:,:3] # right
            ov2 = blk2[:,:overlap_size,:3] # left
        elif orientation == Orientation.BOTTOM_TOP:
            ov1 = np.transpose(blk1[-overlap_size:,:,:3], (1,0,2)) # bottom
            ov2 = np.transpose(blk2[:overlap_size,:,:3], (1,0,2)) # top
        assert ov1.shape == ov2.shape
        return ov1, ov2

    @staticmethod
    def calc_L2_error(blk1, blk2, block_size, overlap_size, orientation):
        ov1, ov2 = Minimum_Cost_Path.get_overlap(blk1, blk2, block_size, overlap_size, orientation)
        L2_error = np.sum((ov1-ov2)**2, axis=2)
        assert (L2_error >= 0).all() == True
        return L2_error

class Image_Quilting:
    def __init__(self, source_image, block_size, overlap_size, number_of_tiles_in_output):
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.number_of_tiles_in_output = number_of_tiles_in_output

        self.image_size = (2*(block_size-overlap_size)) + \
            ((number_of_tiles_in_output-2)*(block_size-2*overlap_size)) + \
            ((number_of_tiles_in_output-1)*overlap_size)
        self.image_channels = source_image.shape[2]

        self.patterns = self.patterns_from_sample(source_image)
        np.random.shuffle(self.patterns)

    def patterns_from_sample(self, source_image):
        patts = set()
        N = self.block_size
        h, w, _ = source_image.shape
        with futures.ProcessPoolExecutor() as pool:
            createPattern = CreatePattern(source_image, N)
            for ident, toappend in pool.map(createPattern,
                                            product(range(0, h-N), range(0, w-N)),
                                            chunksize=w):
                patts.update(toappend)
        return list(patts)

    def join_horizontal_blocks(self, blk1, blk2, path):
        G = [[0,255,0]] # green pixel
        sl1 = blk1[:,-self.overlap_size:]
        sl2 = blk2[:,:self.overlap_size]
        a = path[0]
        if self.debug:
            join_row = lambda i : np.concatenate((sl1[i,:max(0, a-1)], G, sl2[i,max(a, 1):]))
        else:
            join_row = lambda i : np.concatenate((sl1[i,:a], sl2[i,a:]))
        c = join_row(0)
        res = np.zeros((self.block_size, self.overlap_size, self.image_channels), dtype=np.float32)
        res[0,:] = c
        for i in range(1, self.block_size):
            a = path[i]
            c = join_row(i)
            res[i,:] = c
        return np.hstack((res, blk2[:,self.overlap_size:]))

    def join_vertical_blocks(self, blk1, blk2, path):
        G = [[0,255,0]] # green pixel
        sl1 = blk1[-self.overlap_size:,:]
        sl2 = blk2[:self.overlap_size,:]
        a = path[0]
        if self.debug:
            join_col = lambda i : np.concatenate((sl1[:max(0, a-1),i], G, sl2[max(a, 1):,i]))
        else:
            join_col = lambda i : np.concatenate((sl1[:a,i], sl2[a:,i]))
        c = join_col(0)
        res = np.zeros((self.overlap_size, self.block_size, self.image_channels), dtype=np.float32)
        res[:,0] = c
        for i in range(1, self.block_size):
            a = path[i]
            c = join_col(i)
            res[:,i] = c
        return np.vstack((res, blk2[self.overlap_size:,:]))

    def get_random_pattern(self):
        i = np.random.randint(0, high=len(self.patterns))
        return self.patterns[i].data

    def get_best(self, blks, orientation):
        pq = []
        pq_N = 5
        heapq.heapify(pq)
        sample = np.random.choice(self.patterns, size=self.sample_size, replace=False)
        for patt in sample:
            blk = patt.data
            if orientation != Orientation.BOTH:
                l2 = Minimum_Cost_Path.calc_L2_error(blks[0], blk,
                            self.block_size, self.overlap_size, orientation)
                err = l2.sum()
            else:
                l2u = Minimum_Cost_Path.calc_L2_error(blks[0], blk,
                            self.block_size, self.overlap_size, Orientation.BOTTOM_TOP)
                l2l = Minimum_Cost_Path.calc_L2_error(blks[1], blk,
                            self.block_size, self.overlap_size, Orientation.RIGHT_LEFT)
                err = l2u.sum() + l2l.sum()
            pqe = (-err, blk)
            if len(pq) < pq_N:
                heapq.heappush(pq, pqe)
            else:
                try:
                    heapq.heappushpop(pq, pqe)
                except ValueError:
                    # skip errors related to duplicate values
                    print('ValueError!!!:( => {}'.format(pqe))
        idx = np.random.choice(len(pq), 1)[0]
        return pq[idx][1]

    def add_block(self, blk, y, x):
        dx = max(0, x*(self.block_size-self.overlap_size))
        dy = max(0, y*(self.block_size-self.overlap_size))
        self.output_image[dy:dy+self.block_size,dx:dx+self.block_size,:] = blk.copy()

    def get_block(self, y, x):
        dx = max(0, x*(self.block_size-self.overlap_size))
        dy = max(0, y*(self.block_size-self.overlap_size))
        return self.output_image[dy:dy+self.block_size,dx:dx+self.block_size,:].copy()

    def generate(self, sample_size=1, debug=False, show_progress=False):
        self.debug = debug
        self.sample_size = int(np.ceil(len(self.patterns)*sample_size))
        self.output_image = np.zeros((self.image_size, self.image_size, self.image_channels), dtype=np.float32)
        for i in range(self.number_of_tiles_in_output):
            for j in range(self.number_of_tiles_in_output):
                if show_progress: print('\rProgress : (%d,%d)  ' % (i+1,j+1), end = '', flush=True)
                if i == 0 and j == 0:
                    self.output_image[:self.block_size,:self.block_size] = self.get_random_pattern()
                elif i == 0 and j > 0:
                    blk1 = self.get_block(0, j-1) # up
                    blk2 = self.get_best((blk1,), Orientation.RIGHT_LEFT)
                    mcp = Minimum_Cost_Path(blk1, blk2, self.overlap_size, Orientation.RIGHT_LEFT)
                    out = self.join_horizontal_blocks(blk1, blk2, mcp.path)
                    self.add_block(out, i, j)
                elif i > 0 and j == 0:
                    blk1 = self.get_block(i-1, 0) # left
                    blk2 = self.get_best((blk1,), Orientation.BOTTOM_TOP)
                    mcp = Minimum_Cost_Path(blk1, blk2, self.overlap_size, Orientation.BOTTOM_TOP)
                    out = self.join_vertical_blocks(blk1, blk2, mcp.path)
                    self.add_block(out, i, j)
                elif i > 0 and j > 0:
                    blk1 = self.get_block(i-1, j) # up
                    blk2 = self.get_block(i, j-1) # left
                    blk3 = self.get_best((blk1, blk2), Orientation.BOTH)
                    mcp1 = Minimum_Cost_Path(blk1, blk3, self.overlap_size, Orientation.BOTTOM_TOP)
                    mcp2 = Minimum_Cost_Path(blk2, blk3, self.overlap_size, Orientation.RIGHT_LEFT)
                    assert mcp1 != mcp2
                    out1 = self.join_vertical_blocks(blk1, blk3, mcp1.path)
                    out2 = self.join_horizontal_blocks(blk2, out1, mcp2.path)
                    out1.shape == out2.shape
                    self.add_block(out2, i, j)

        return self.output_image

block_size = 64
overlap_size = block_size//6
number_of_tiles_in_output = 10 # output image widht in tiles

def load_source_image(source_path):
    I1 = plt.imread(source_path)

    print('I1.shape={}, I1.dtype={}, I1.max={}, I1.min={}'.format(
          I1.shape, I1.dtype, I1.max(), I1.min()))

#    I1 = (I1-I1.min()) / (I1.max()-I1.min())
#    I1 = I1.astype(np.float32)
    assert I1.min() >= 0. and I1.max() <= 1.
#    plt.imshow(I1)
#    plt.show()
#
#    print('I1.shape={}, I1.dtype={}, I1.max={}, I1.min={}'.format(
#          I1.shape, I1.dtype, I1.max(), I1.min()))

    return I1

if __name__ == '__main__':
    np.random.seed(opt.manualSeed)
    source_image = load_source_image(opt.inputImage)
    iq = Image_Quilting(source_image, block_size, overlap_size, number_of_tiles_in_output)
    print('Number of patterns = {}'.format(len(iq.patterns)))
    for i in range(5):
        output_image = iq.generate(sample_size=1, debug=False, show_progress=True)
        plt.imsave(os.path.join(opt.outputFolder, 'iq_{}x{}_{}.png').
                   format(number_of_tiles_in_output, number_of_tiles_in_output, i), output_image)