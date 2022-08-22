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
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

controlm = ['y','o']
TuningType = 'both'
colors = ['dimgray', 'crimson', 'green', 'darkorange']

N_cells = 1000.
N_HL5PN = int(0.70*N_cells)

dt = 0.025
tstop = 7000

transient = 2000
stimtime = 6500 # Absolute time during stimulation that is set for stimtime
stimbegin = 3 # Time (ms) after stimtime where stimulation begins (i.e. the system delay)
stimend = 103 # Stimbegin + duration of stimulus

binsize = 5
gstart = -200
gstop = 200
numbins = int((gstop - gstart)/binsize)

startslice = stimtime+stimbegin
endslice = stimtime+stimend
totalstim = endslice-startslice
tvec = np.arange(startslice,endslice+1,dt,dtype=np.float64)

synnums = [85,95]
N_seeds = 80
rndinds = np.linspace(1,N_seeds,N_seeds, dtype=int)

for synnum in synnums:
	for control in controlm:
		path = control + '_' + str(synnum)+ '/HL5netLFPy/Circuit_output/'
		hist_counts = [[] for _ in colors]
		for idx in range(0,len(rndinds)):
			print('Seed #'+str(idx))
			temp_sn = np.load(path + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
			
			for pop in range(0,len(colors)):
				SPIKES = [x for _,x in sorted(zip(temp_sn.item()['gids'][pop],temp_sn.item()['times'][pop]))]
				popspikes = list(itertools.chain.from_iterable(SPIKES))
				popspikes = [i2-stimtime for i2 in popspikes if i2 > transient]
				counts, bin_edges = np.histogram(popspikes,bins=numbins,range=(gstart,gstop))
				hist_counts[pop].append(counts)
		
		fig, axarr = plt.subplots(nrows=len(colors),ncols=1,figsize=(8,12),sharey=False,sharex=True)
		fig.text(0, 0.5, 'Spike Rate (Hz)', va='center', rotation='vertical')
		for pop in range(0,len(colors)):
			popsize = len(temp_sn.item()['gids'][pop])
			meancounts_rate = (np.mean(hist_counts[pop],axis=0)/popsize)/(binsize/1000)
			sdcounts_rate = (np.std(hist_counts[pop],axis=0)/popsize)/(binsize/1000)
			xinds = bin_edges[:-1]+binsize/2
			axarr[pop].bar(xinds,meancounts_rate,width=binsize,yerr=sdcounts_rate,color=colors[pop])
			axarr[pop].set_xlim(gstart,gstop)
			axarr[0].set_ylim(0,30) # for SST axis
			axarr[1].set_ylim(0,30) # for SST axis
			ylims = axarr[pop].get_ylim()
			axarr[pop].plot(np.array([0,0]),ylims,ls='dashed',c='r')
			axarr[pop].set_ylim(ylims)
		axarr[-1].set_xlabel('Time (ms)')
		fig.gca().set_ylim(bottom=0)
		fig.tight_layout()
		fig.savefig('figs_tuning/PeriStimulusHist_'+str(totalstim)+'ms_'+str(synnum)+control+'.png',bbox_inches='tight', dpi=300, transparent=True)
		plt.close(fig)
