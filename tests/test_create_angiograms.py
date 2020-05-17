from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob

# PARAMETERS FOR RAW DATA SOURCE
filename = './octa_test_set.unp'
n_vol = 1
n_slow = 4
n_repeats = 5
n_fast = 2500
n_skip = 500
n_depth = 1536
bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

# PROCESSING PARAMETERS
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]

fft_oversampling_size = 4096
bscan_z1 = 3147
bscan_z2 = -40
bscan_x1 = 0
bscan_x2 = -100

# In this section, we will load one set of repeats and arrange them in a 3D array
# to be bulk-motion corrected
for frame_index in range(4):
    print(frame_index)
    frame = src.get_frame(frame_index)
    frame = blob.dc_subtract(frame)
    frame = blob.k_resample(frame,mapping_coefficients)
    frame = blob.dispersion_compensate(frame,dispersion_coefficients)
    frame = blob.gaussian_window(frame,0.9)
    bscan = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2)

    stack_complex = blob.reshape_repeats(bscan,n_repeats,x1=bscan_x1,x2=bscan_x2)
    phase_variance,log_bscan = blob.make_angiogram(stack_complex)
    plt.figure()
    plt.imshow(phase_variance,aspect='auto')
    
plt.show()
