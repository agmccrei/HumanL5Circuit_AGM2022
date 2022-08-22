################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
from __future__ import division
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.fftpack
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import itertools

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

controlm = ['y','o']
TuningType = 'both'
colors = ['dimgray', 'crimson', 'green', 'darkorange']

dt = 0.025
tstop = 7000

stimtime = 6500 # Absolute time during stimulation that is set for stimtime
stimbegin = 3 # Time (ms) after stimtime where stimulation begins (i.e. the system delay)
stimend = 103 # Stimbegin + duration of stimulus
startslice = stimtime+stimbegin
endslice = stimtime+stimend
totalstim = endslice-startslice
tvec_psp = np.arange(0,endslice-startslice+1,dt,dtype=np.float64)
tvec = np.arange(startslice,endslice+1,dt,dtype=np.float64)

# Time constants for PSP convolution
taur_AMPA = 0.5
taud_AMPA = 3
taur_NMDA = 2
taud_NMDA = 65
psp_AMPA = np.exp(-(tvec_psp/taud_AMPA))-np.exp(-(tvec_psp/taur_AMPA))
psp_NMDA = np.exp(-(tvec_psp/taud_NMDA))-np.exp(-(tvec_psp/taur_NMDA))
psp_AMPANMDA = np.mean([psp_AMPA,psp_NMDA],axis=0)

fig, axarr = plt.subplots(1,1)
axarr.plot(tvec_psp,psp_AMPA,'k')
axarr.set_xlim(tvec_psp[0]-0.5,tvec_psp[-1])
axarr.spines['right'].set_visible(False)
axarr.spines['left'].set_visible(False)
axarr.spines['top'].set_visible(False)
axarr.spines['bottom'].set_visible(False)
fig.savefig('figs_readout/PSP_AMPA.png',dpi=300,transparent=True)

fig, axarr = plt.subplots(1,1)
axarr.plot(tvec_psp,psp_NMDA,'k')
axarr.set_xlim(tvec_psp[0]-0.5,tvec_psp[-1])
axarr.spines['right'].set_visible(False)
axarr.spines['left'].set_visible(False)
axarr.spines['top'].set_visible(False)
axarr.spines['bottom'].set_visible(False)
fig.savefig('figs_readout/PSP_NMDA.png',dpi=300,transparent=True)

fig, axarr = plt.subplots(1,1)
axarr.plot(tvec_psp,psp_AMPANMDA,'k')
axarr.set_xlim(tvec_psp[0]-0.5,tvec_psp[-1])
axarr.spines['right'].set_visible(False)
axarr.spines['left'].set_visible(False)
axarr.spines['top'].set_visible(False)
axarr.spines['bottom'].set_visible(False)
fig.savefig('figs_readout/PSP_AMPANMDA.png',dpi=300,transparent=True)

synnums = np.array([85,95])
N_seeds = 80
rndinds = np.linspace(1,N_seeds,N_seeds, dtype=int)

for control in controlm:
	if control == 'y': colc = 'k'
	if control == 'o': colc = 'r'
	normalpath = [control + '_' + str(sn)+ '/HL5netLFPy/Circuit_output/' for sn in synnums]
		
	for count, path in enumerate(normalpath):
		# integ_allseeds = [[] for _ in range(0,len(rndinds))]
		integ_area_allseeds = [[] for _ in range(0,len(rndinds))]
		integ_max_allseeds = [[] for _ in range(0,len(rndinds))]
		integ_num_allseeds = [[] for _ in range(0,len(rndinds))]
		integ_rate_allseeds = [[] for _ in range(0,len(rndinds))]
		
		for idx in range(0,len(rndinds)):
			print('Seed #'+str(idx))
			temp_sn = np.load(path + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
			
			# Only analyze PNs here
			SPIKES1 = [x for _,x in sorted(zip(temp_sn.item()['gids'][0],temp_sn.item()['times'][0]))]
			for j in range(len(SPIKES1)):
				spikebinvec_PN = np.zeros(len(tvec))
				tv = np.around(np.array(SPIKES1[j][(SPIKES1[j]>startslice) & (SPIKES1[j]<endslice)]),3)
				for x in tv: spikebinvec_PN[np.where(abs(tvec - x)<1e-8)] = spikebinvec_PN[np.where(abs(tvec - x)<1e-8)] + 1
				# scipy convolve method with fft is faster than numpy.convolve
				# area array size [seed index][stim index]
				integ = ss.convolve(spikebinvec_PN,psp_AMPANMDA,method='fft')
				integ = integ[0:len(tvec)]
				# integ_allseeds[idx].append(integ)
				integ_area = np.trapz(integ,x=tvec)
				integ_area_allseeds[idx].append(integ_area)
				integ_max = np.max(integ)
				integ_max_allseeds[idx].append(integ_max)
				integ_num_allseeds[idx].append(len(tv))
				integ_rate_allseeds[idx].append(len(tv)/totalstim)
		
		# Save analyzed data into respective grouping
		if ((count == 0) & (control == 'y')):
			# inte_y_angel1 = integ_allseeds
			area_y_angel1 = integ_area_allseeds
			max_y_angel1 = integ_max_allseeds
			num_y_angel1 = integ_num_allseeds
			rate_y_angel1 = integ_rate_allseeds
		elif ((count == 1) & (control == 'y')):
			# inte_y_angel2 = integ_allseeds
			area_y_angel2 = integ_area_allseeds
			max_y_angel2 = integ_max_allseeds
			num_y_angel2 = integ_num_allseeds
			rate_y_angel2 = integ_rate_allseeds
		elif ((count == 0) & (control == 'o')):
			# inte_o_angel1 = integ_allseeds
			area_o_angel1 = integ_area_allseeds
			max_o_angel1 = integ_max_allseeds
			num_o_angel1 = integ_num_allseeds
			rate_o_angel1 = integ_rate_allseeds
		elif ((count == 1) & (control == 'o')):
			# inte_o_angel2 = integ_allseeds
			area_o_angel2 = integ_area_allseeds
			max_o_angel2 = integ_max_allseeds
			num_o_angel2 = integ_num_allseeds
			rate_o_angel2 = integ_rate_allseeds

np.save('output_readout/area'+str(totalstim)+'y_a1.npy',area_y_angel1)
np.save('output_readout/area'+str(totalstim)+'o_a1.npy',area_o_angel1)
np.save('output_readout/area'+str(totalstim)+'y_a2.npy',area_y_angel2)
np.save('output_readout/area'+str(totalstim)+'o_a2.npy',area_o_angel2)

np.save('output_readout/max'+str(totalstim)+'y_a1.npy',max_y_angel1)
np.save('output_readout/max'+str(totalstim)+'o_a1.npy',max_o_angel1)
np.save('output_readout/max'+str(totalstim)+'y_a2.npy',max_y_angel2)
np.save('output_readout/max'+str(totalstim)+'o_a2.npy',max_o_angel2)

np.save('output_readout/num'+str(totalstim)+'y_a1.npy',num_y_angel1)
np.save('output_readout/num'+str(totalstim)+'o_a1.npy',num_o_angel1)
np.save('output_readout/num'+str(totalstim)+'y_a2.npy',num_y_angel2)
np.save('output_readout/num'+str(totalstim)+'o_a2.npy',num_o_angel2)

np.save('output_readout/rate'+str(totalstim)+'y_a1.npy',rate_y_angel1)
np.save('output_readout/rate'+str(totalstim)+'o_a1.npy',rate_o_angel1)
np.save('output_readout/rate'+str(totalstim)+'y_a2.npy',rate_y_angel2)
np.save('output_readout/rate'+str(totalstim)+'o_a2.npy',rate_o_angel2)

# zero out values below
area_y_angel1 = np.array(area_y_angel1)
area_y_angel2 = np.array(area_y_angel2)
area_o_angel1 = np.array(area_o_angel1)
area_o_angel2 = np.array(area_o_angel2)

area_y_angel1[area_y_angel1<np.percentile(area_y_angel1,90)] = 0
area_y_angel2[area_y_angel2<np.percentile(area_y_angel2,90)] = 0
area_o_angel1[area_o_angel1<np.percentile(area_o_angel1,90)] = 0
area_o_angel2[area_o_angel2<np.percentile(area_o_angel2,90)] = 0

area_y_angel1 = area_y_angel1.tolist()
area_y_angel2 = area_y_angel2.tolist()
area_o_angel1 = area_o_angel1.tolist()
area_o_angel2 = area_o_angel2.tolist()

np.save('output_readout/areahf'+str(totalstim)+'y_a1.npy',area_y_angel1)
np.save('output_readout/areahf'+str(totalstim)+'o_a1.npy',area_o_angel1)
np.save('output_readout/areahf'+str(totalstim)+'y_a2.npy',area_y_angel2)
np.save('output_readout/areahf'+str(totalstim)+'o_a2.npy',area_o_angel2)

max_y_angel1 = np.array(max_y_angel1)
max_y_angel2 = np.array(max_y_angel2)
max_o_angel1 = np.array(max_o_angel1)
max_o_angel2 = np.array(max_o_angel2)

max_y_angel1[max_y_angel1<np.percentile(max_y_angel1,90)] = 0
max_y_angel2[max_y_angel2<np.percentile(max_y_angel2,90)] = 0
max_o_angel1[max_o_angel1<np.percentile(max_o_angel1,90)] = 0
max_o_angel2[max_o_angel2<np.percentile(max_o_angel2,90)] = 0

max_y_angel1 = max_y_angel1.tolist()
max_y_angel2 = max_y_angel2.tolist()
max_o_angel1 = max_o_angel1.tolist()
max_o_angel2 = max_o_angel2.tolist()

np.save('output_readout/maxhf'+str(totalstim)+'y_a1.npy',max_y_angel1)
np.save('output_readout/maxhf'+str(totalstim)+'o_a1.npy',max_o_angel1)
np.save('output_readout/maxhf'+str(totalstim)+'y_a2.npy',max_y_angel2)
np.save('output_readout/maxhf'+str(totalstim)+'o_a2.npy',max_o_angel2)

# Below creates several GBs of data files
# np.save('output_readout/integy_a1.npy',inte_y_angel1)
# np.save('output_readout/intego_a1.npy',inte_o_angel1)
# np.save('output_readout/integy_a2.npy',inte_y_angel2)
# np.save('output_readout/intego_a2.npy',inte_o_angel2)
