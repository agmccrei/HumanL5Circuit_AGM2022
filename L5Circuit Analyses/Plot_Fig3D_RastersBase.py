################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
from __future__ import division
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.fftpack
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import itertools
import pandas as pd

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 26}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':18})

controlm = ['y','o']
TuningType = 'both'
colors = ['dimgray', 'crimson', 'green', 'darkorange']

N_cells = 1000.
N_HL5PN = int(0.70*N_cells)

dt = 0.025
tstop = 7000

transient = 2000
stimtime = transient # Absolute time during stimulation that is set for stimtime
stimbegin = 0 # Time (ms) after stimtime where stimulation begins (i.e. the system delay)
stimend = 400 # Stimbegin + duration of stimulus

binsize = 3
gstart = 0
gstop = 400
numbins = int((gstop - gstart)/binsize)

startslice = stimtime+stimbegin
endslice = stimtime+stimend
totalstim = endslice-startslice
tvec = np.arange(startslice,endslice+1,dt,dtype=np.float64)

synnums = [85,95]
N_seeds = 80
rndinds = np.linspace(1,N_seeds,N_seeds, dtype=int)

def plot_raster(SPIKES,gstart=gstart,gstop=gstop,N_cells=N_cells,stimtime=stimtime):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	pop_colors = {'HL5PN1':'k', 'HL5MN1':'crimson', 'HL5BN1':'green', 'HL5VN1':'darkorange'}
	popnames = ['HL5PN1', 'HL5MN1', 'HL5BN1', 'HL5VN1']
	fig_t = plt.figure(figsize=(5, 8))
	ax1_t =fig_t.add_subplot(111)
	# ax1_t.plot(np.array([0,0]),np.array([0,N_cells]),ls='dashed',c='r')
	
	for name, spts, gids in zip(popnames, SPIKES['times'], SPIKES['gids']):
		t = []
		g = []
		ind=0
		for spt, gid in zip(spts, gids):
			t = np.r_[t, spt]
			g = np.r_[g, np.zeros(spt.size)+gid]
			ax1_t.plot(t[t >= transient]-stimtime, g[t >= transient], '|', color=pop_colors[name])
			ind+=1
		ax1_t.set_ylim(0,N_cells)
		ax1_t.set_xlim(gstart,gstop)
		ax1_t.set_xlabel('Time (ms)')
		ax1_t.set_ylabel('Neuron Index')
	
	return fig_t

for synnum in synnums:
	for control in controlm:
		path = control + '_' + str(synnum)+ '/HL5netLFPy/Circuit_output/'
		for idx in range(0,len(rndinds)):
			print('Seed #'+str(idx))
			temp_sn = np.load(path + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
			
			fig_raster = plot_raster(temp_sn.item(),gstart=gstart,gstop=gstop,N_cells=N_cells,stimtime=stimtime)
			fig_raster.savefig('figs_rasters_base/raster_seed'+str(idx)+'_angle_'+str(synnum)+control+'.png',bbox_inches='tight', dpi=300, transparent=True)
			plt.close(fig_raster)
