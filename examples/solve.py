#!/usr/bin/env python

from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

import sys
caffe_root = '../../' 
sys.path.insert(0, caffe_root + 'python')
import caffe

import scipy.io as scio

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

#base_weights = 'snapshots/final.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver('solver.prototxt')
# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

solver.net.copy_from(base_weights)
solver.step(80000)

