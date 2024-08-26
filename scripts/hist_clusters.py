import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool
import time

def cluster_diams(ids, nclusters):
    nrows, ncols = ids.shape
    nsites = nrows*ncols
    diams = np.ones(nclusters)
    for a in range(nsites):
        for b in range(a+1, nsites):
            (i, j) = (a // ncols, a % ncols)
            (k, l) = (b // ncols, b % ncols)
            aid = ids[i, j]
            if aid == -1 or aid != ids[k, l]:
                continue
            d = (abs(i - k) + 1)**2 + (abs(j - l) + 1)**2
            cidx = aid-2 # id in cluster array is 2 less since cluster indexing starts at 2
            if d > diams[cidx]:
                diams[cidx] = d
    return list(map(math.sqrt, diams))

parser = argparse.ArgumentParser(description='Create cluster property histograms from cluster microstates')
parser.add_argument('--indir', type=str, default="temp", help='input directory')
parser.add_argument('--prefix', type=str, default="fig", help='outfile prefix')
parser.add_argument('--ext', type=str, default=".png", help='file extension for figure files')
parser.add_argument('--bins', default=None, help='int or sequence or str, optional (number of bins, bin boundaries, etc., see matplotlib docs)')
parser.add_argument('--orientation', type=str, default='vertical', help='orientation of histogram bars')
parser.add_argument('--range', default=None, help='lower and upper range of the bins')
parser.add_argument('--density', default=False, action="store_true", help='normalize bins')
parser.add_argument('--histtype', type=str, default='bar', help='type of histogram to draw')
parser.add_argument('--log', default=False, action="store_true", help='log scale for histogram axis')
parser.add_argument('--timeout', default=None, help='maximum time to compute histograms')
parser.add_argument('--minsize', type=int, default=1, help='minimum cluster size to consider')

namespace_args = parser.parse_args()
args = vars(namespace_args)
d = args.pop('indir')
prefix = args.pop('prefix')
ext = args.pop('ext')
timeout = float(args.pop('timeout'))
minsize = args.pop('minsize')
print(d, prefix, ext, timeout, args)

clsizes = []
cleccs = []
start = time.time()
for f in os.listdir(d):
    if timeout != None and time.time() - start > timeout:
        print('timeout! ', time.time() - start, timeout, time.time() - start > timeout)
        break
    fname = os.path.join(d, f)
    print(fname)
    if f.startswith('cluster-ids') and f.endswith('.csv'):
        ids = np.genfromtxt(fname, delimiter=',', dtype=int)
        nclusters = int(np.max(ids))-1
        print(nclusters)
        clsizes_loc = [np.sum(ids == i) for i in range(2, nclusters+2)] # maximum cluster id is the nclusters+1, so iterate until nclusters+2
        cdiams = cluster_diams(ids, nclusters)
        cleccs_loc = [math.sqrt(1 - (4*A / (math.pi*d**2))**2) if A > 1 else 0.0 for (A, d) in zip(clsizes_loc, cdiams)]
        for (sz, ec) in zip(clsizes_loc, cleccs_loc):
            if sz >= minsize:
                clsizes.append(sz)
                cleccs.append(ec)

print('clsizes = ', clsizes)
print('cleccs = ', cleccs)

plt.hist(clsizes, **args)
plt.savefig(prefix + '_shist' + ext)
plt.clf()

plt.hist(cleccs, **args)
plt.savefig(prefix + '_ehist' + ext)

np.savetxt(prefix + '_datahist.csv', np.column_stack((clsizes, cleccs)), delimiter=',')
