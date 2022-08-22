#================================================================================
#= Import
#================================================================================
import os
import time
tic = time.perf_counter()
from os.path import join
import sys
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.fftpack
from scipy import signal as ss
from scipy import stats as st
from mpi4py import MPI
import math
import neuron
from neuron import h, gui
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode
from net_params import *
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

plotnetworksomas = True
plotrasterandrates = True
plotephistimseriesandPSD = True
plotsomavs = True # Specify cell indices to plot in 'cell_indices_to_plot' - Note: plotting too many cells can randomly cause TCP connection errors
plotsynlocs = False

#================================================================================
#= Analysis
#================================================================================
#===============================
#= Analysis Parameters
#===============================
transient = 2000 #used for plotting and analysis

radii = [79000., 80000., 85000., 90000.] #4sphere model
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L5_pos = np.array([0., 0., 77200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

sampling_rate = (1/0.025)*1000
nperseg = 100000#int(sampling_rate/2)
t1 = int(transient/0.025)
#===============================

def bandPassFilter(signal,low=.1, high=100.,order = 2):
	z, p, k = ss.butter(order, [low,high],btype='bandpass',fs=sampling_rate,output='zpk')
	sos = ss.zpk2sos(z, p, k)
	y = ss.sosfiltfilt(sos, signal)
	# b, a = ss.butter(order, [low,high],btype='bandpass',fs=sampling_rate)
	# y = ss.filtfilt(b, a, signal)
	return y

#================================================================================
#= Plotting
#================================================================================
#===============================
#= Plotting Parameters
#===============================
pop_colors = {'HL5PN1':'k', 'HL5MN1':'crimson', 'HL5BN1':'green', 'HL5VN1':'darkorange'}
popnames = ['HL5PN1', 'HL5MN1', 'HL5BN1', 'HL5VN1']
poplabels = ['PN', 'MN', 'BN', 'VN']
#===============================


# Plot soma positions
def plot_network_somas(OUTPUTPATH):
	filename = os.path.join(OUTPUTPATH,'cell_positions_and_rotations.h5')
	popDataArray = {}
	popDataArray[popnames[0]] = pd.read_hdf(filename,popnames[0])
	popDataArray[popnames[0]] = popDataArray[popnames[0]].sort_values('gid')
	popDataArray[popnames[1]] = pd.read_hdf(filename,popnames[1])
	popDataArray[popnames[1]] = popDataArray[popnames[1]].sort_values('gid')
	popDataArray[popnames[2]] = pd.read_hdf(filename,popnames[2])
	popDataArray[popnames[2]] = popDataArray[popnames[2]].sort_values('gid')
	popDataArray[popnames[3]] = pd.read_hdf(filename,popnames[3])
	popDataArray[popnames[3]] = popDataArray[popnames[3]].sort_values('gid')
	
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=5)
	for pop in popnames:
		for i in range(0,len(popDataArray[pop]['gid'])):
			ax.scatter(popDataArray[pop]['x'][i],popDataArray[pop]['y'][i],popDataArray[pop]['z'][i], c=pop_colors[pop], s=5)
			ax.set_xlim([-300, 300])
			ax.set_ylim([-300, 300])
			ax.set_zlim([-2600, -1700])
	return fig

# Plot spike raster plots & spike rates
def plot_raster_and_rates(SPIKES,tstart_plot,tstop_plot,popnames,N_cells,network,OUTPUTPATH,GLOBALSEED,stimtime=network.tstop):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	fig = plt.figure(figsize=(10, 8))
	ax1 =fig.add_subplot(111)
	ax1.plot(np.array([stimtime,stimtime]),np.array([0,N_cells]),ls='dashed',c='r')
	
	for name, spts, gids in zip(popnames, SPIKES['times'], SPIKES['gids']):
		t = []
		g = []
		for spt, gid in zip(spts, gids):
			t = np.r_[t, spt]
			g = np.r_[g, np.zeros(spt.size)+gid]
			ax1.plot(t[t >= transient], g[t >= transient], '|', color=pop_colors[name])
		ax1.set_ylim(0,N_cells)
		# halftime = 750
		# plt1 = int(tstart_plot+((tstop_plot-tstart_plot)/2)-halftime)
		# plt2 = int(tstart_plot+((tstop_plot-tstart_plot)/2)+halftime)
		ax1.set_xlim(tstart_plot,tstop_plot)
		ax1.set_xlabel('Time (ms)')
		ax1.set_ylabel('Cell Number')
	
	PN = np.zeros(len(SPIKES['times'][0]))
	PN2 = np.zeros(len(SPIKES['times'][0]))
	MN = np.zeros(len(SPIKES['times'][1]))
	MN2 = np.zeros(len(SPIKES['times'][1]))
	BN = np.zeros(len(SPIKES['times'][2]))
	BN2 = np.zeros(len(SPIKES['times'][2]))
	VN = np.zeros(len(SPIKES['times'][3]))
	VN2 = np.zeros(len(SPIKES['times'][3]))
	SPIKE_list = [PN ,MN, BN, VN]
	SPIKE_liststim = [PN2 ,MN2, BN2, VN2]
	SPIKE_list_se = []
	SPIKE_liststim_se = []
	SPIKE_MEANS = []
	SPIKE_MEANSstim = []
	SPIKE_STDEV = []
	SPIKE_STDEVstim = []
	SPIKE_MEANS_se = []
	SPIKE_MEANSstim_se = []
	SPIKE_STDEV_se = []
	SPIKE_STDEVstim_se = []
	SILENT_list = np.zeros(len(SPIKE_list))
	SILENT_liststim = np.zeros(len(SPIKE_list))
	PERCENT_SILENT = []
	PERCENT_SILENTstim = []
	for i, pop in enumerate(network.populations):
		for j in range(len(SPIKES['times'][i])):
			scount = SPIKES['times'][i][j][(SPIKES['times'][i][j]>transient) & (SPIKES['times'][i][j]<=stimtime)]
			scount2 = SPIKES['times'][i][j][(SPIKES['times'][i][j]>(stimtime+5)) & (SPIKES['times'][i][j]<=(stimtime+105))]
			Hz = (scount.size)/((int(stimtime)-transient)/1000)
			Hz2 = (scount2.size)/((int(stimtime+105)-int(stimtime+5))/1000)
			SPIKE_list[i][j] = Hz
			SPIKE_liststim[i][j] = Hz2
			if Hz <= 0.2:
				SILENT_list[i] += 1
				SILENT_liststim[i] += 1
		print(SPIKE_list[i])
		print(SPIKE_liststim[i])
		SPIKE_list_se.append(SPIKE_list[i][SPIKE_list[i]>0.2])
		SPIKE_liststim_se.append(SPIKE_liststim[i][SPIKE_list[i]>0.2])
		PERCENT_SILENT.append((SILENT_list[i]/len(SPIKES['times'][i]))*100)
		PERCENT_SILENTstim.append((SILENT_liststim[i]/len(SPIKES['times'][i]))*100)
		print('%',poplabels[i],' Silent: ',str(PERCENT_SILENT[i]))
		SPIKE_MEANS.append(np.mean(SPIKE_list[i]))
		SPIKE_MEANSstim.append(np.mean(SPIKE_liststim[i]))
		SPIKE_STDEV.append(np.std(SPIKE_list[i]))
		SPIKE_STDEVstim.append(np.std(SPIKE_liststim[i]))
		SPIKE_MEANS_se.append(np.mean(SPIKE_list_se[i]))
		SPIKE_MEANSstim_se.append(np.mean(SPIKE_liststim_se[i]))
		SPIKE_STDEV_se.append(np.std(SPIKE_list_se[i]))
		SPIKE_STDEVstim_se.append(np.std(SPIKE_liststim_se[i]))
	
	meanstdevstr1 = '\n' + str(np.around(SPIKE_MEANS[0], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[0], decimals=2))
	meanstdevstr2 = '\n' + str(np.around(SPIKE_MEANS[1], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[1], decimals=2))
	meanstdevstr3 = '\n' + str(np.around(SPIKE_MEANS[2], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[2], decimals=2))
	meanstdevstr4 = '\n' + str(np.around(SPIKE_MEANS[3], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[3], decimals=2))
	names = [poplabels[0]+meanstdevstr1,poplabels[1]+meanstdevstr2,poplabels[2]+meanstdevstr3,poplabels[3]+meanstdevstr4]

	meanstdevstr1stim = '\n' + str(np.around(SPIKE_MEANSstim[0], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim[0], decimals=2))
	meanstdevstr2stim = '\n' + str(np.around(SPIKE_MEANSstim[1], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim[1], decimals=2))
	meanstdevstr3stim = '\n' + str(np.around(SPIKE_MEANSstim[2], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim[2], decimals=2))
	meanstdevstr4stim = '\n' + str(np.around(SPIKE_MEANSstim[3], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim[3], decimals=2))
	namesstim = [poplabels[0]+meanstdevstr1stim,poplabels[1]+meanstdevstr2stim,poplabels[2]+meanstdevstr3stim,poplabels[3]+meanstdevstr4stim]
	
	meanstdevstr1_se = '\n' + str(np.around(SPIKE_MEANS_se[0], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV_se[0], decimals=2))
	meanstdevstr2_se = '\n' + str(np.around(SPIKE_MEANS_se[1], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV_se[1], decimals=2))
	meanstdevstr3_se = '\n' + str(np.around(SPIKE_MEANS_se[2], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV_se[2], decimals=2))
	meanstdevstr4_se = '\n' + str(np.around(SPIKE_MEANS_se[3], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV_se[3], decimals=2))
	names_se = [poplabels[0]+meanstdevstr1_se,poplabels[1]+meanstdevstr2_se,poplabels[2]+meanstdevstr3_se,poplabels[3]+meanstdevstr4_se]
	
	meanstdevstr1stim_se = '\n' + str(np.around(SPIKE_MEANSstim_se[0], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim_se[0], decimals=2))
	meanstdevstr2stim_se = '\n' + str(np.around(SPIKE_MEANSstim_se[1], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim_se[1], decimals=2))
	meanstdevstr3stim_se = '\n' + str(np.around(SPIKE_MEANSstim_se[2], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim_se[2], decimals=2))
	meanstdevstr4stim_se = '\n' + str(np.around(SPIKE_MEANSstim_se[3], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEVstim_se[3], decimals=2))
	namesstim_se = [poplabels[0]+meanstdevstr1stim_se,poplabels[1]+meanstdevstr2stim_se,poplabels[2]+meanstdevstr3stim_se,poplabels[3]+meanstdevstr4stim_se]
	
	Hzs_mean = np.array(SPIKE_MEANS)
	Hzs_mean_se = np.array(SPIKE_MEANS_se)
	Hzs_meanstim = np.array(SPIKE_MEANSstim)
	Hzs_meanstim_se = np.array(SPIKE_MEANSstim_se)
	np.savetxt(os.path.join(OUTPUTPATH,'spikerates_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),Hzs_mean)
	np.savetxt(os.path.join(OUTPUTPATH,'spikerates_SilentNeuronsExcluded_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),Hzs_mean_se)
	np.savetxt(os.path.join(OUTPUTPATH,'spikeratesstim_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),Hzs_meanstim)
	np.savetxt(os.path.join(OUTPUTPATH,'spikeratesstim_SilentNeuronsExcluded_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),Hzs_meanstim_se)
	np.savetxt(os.path.join(OUTPUTPATH,'spikerates_PERCENTSILENT_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),PERCENT_SILENT)
	np.savetxt(os.path.join(OUTPUTPATH,'spikerates_PERCENTSILENTstim_Seed' + str(int(GLOBALSEED)) + 'StimSeed' + str(int(STIMSEED)) + '.txt'),PERCENT_SILENTstim)
	w = 0.8
	
	fig2 = plt.figure(figsize=(10, 8))
	ax2 = fig2.add_subplot(111)
	ax2.bar(x = [0, 1, 2, 3],
	height=[pop for pop in SPIKE_MEANS],
	yerr=[np.std(pop) for pop in SPIKE_list],
	capsize=12,
	width=w,
	tick_label=names,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax2.set_ylabel('Spike Frequency (Hz)')
	ax2.grid(False)
	
	fig3 = plt.figure(figsize=(10, 8))
	ax3 = fig3.add_subplot(111)
	ax3.bar(x = [0, 1, 2, 3],
	height=[pop for pop in SPIKE_MEANS_se],
	yerr=[np.std(pop) for pop in SPIKE_list_se],
	capsize=12,
	width=w,
	tick_label=names_se,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax3.set_ylabel('Spike Frequency (Hz)')
	ax3.grid(False)
	
	barWidth = 0.3
	r1 = np.arange(len(SPIKE_MEANS))
	r2 = [x + barWidth for x in r1]
	fig4 = plt.figure(figsize=(10, 8))
	ax4 = fig4.add_subplot(111)
	ax4.bar(x = r1,
	height=[pop for pop in SPIKE_MEANS],
	yerr=[np.std(pop)/np.sqrt(len(pop)) for pop in SPIKE_list],
	capsize=12,
	width=barWidth,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax4.bar(x = r2,
	height=[pop for pop in SPIKE_MEANSstim],
	yerr=[np.std(pop)/np.sqrt(len(pop)) for pop in SPIKE_liststim],
	capsize=12,
	width=barWidth,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax4.set_xticks([0+barWidth/2,1+barWidth/2,2+barWidth/2,3+barWidth/2])
	ax4.set_xticklabels(['PN','MN','BN','VN'])
	ax4.set_ylabel('Spike Frequency (Hz)')
	ax4.grid(False)
	
	barWidth = 0.3
	r1 = np.arange(len(SPIKE_MEANS_se))
	r2 = [x + barWidth for x in r1]
	fig5 = plt.figure(figsize=(10, 8))
	ax5 = fig5.add_subplot(111)
	ax5.bar(x = r1,
	height=[pop for pop in SPIKE_MEANS_se],
	yerr=[np.std(pop)/np.sqrt(len(pop)) for pop in SPIKE_list_se],
	capsize=12,
	width=barWidth,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax5.bar(x = r2,
	height=[pop for pop in SPIKE_MEANSstim_se],
	yerr=[np.std(pop)/np.sqrt(len(pop)) for pop in SPIKE_liststim_se],
	capsize=12,
	width=barWidth,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax5.set_xticks([0+barWidth/2,1+barWidth/2,2+barWidth/2,3+barWidth/2])
	ax5.set_xticklabels(['PN','MN','BN','VN'])
	ax5.set_ylabel('Spike Frequency (Hz)')
	ax5.grid(False)
	
	return fig, fig2, fig3, fig4, fig5

# Plot spike time histograms
def plot_spiketimehists(SPIKES,network,gstart=transient,gstop=network.tstop,stimtime=0,binsize=10):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	numbins = int((gstop - gstart)/binsize)
	fig, axarr = plt.subplots(len(colors),1)
	for i, pop in enumerate(network.populations):
		popspikes = list(itertools.chain.from_iterable(SPIKES['times'][i]))
		popspikes = [i2-stimtime for i2 in popspikes if i2 > transient]
		axarr[i].hist(popspikes,bins=numbins,color=colors[i],linewidth=0,edgecolor='none',range=(gstart,gstop))
		axarr[i].set_xlim(gstart,gstop)
		if i < len(colors)-1:
			axarr[i].set_xticks([])
	axarr[-1:][0].set_xlabel('Time (ms)')
	
	return fig

# Plot spike vector PSDs
def plot_spikevecPSDs(SPIKES,network,stimtime=network.tstop):
	t1 = int(transient/network.dt)
	t2 = int(stimtime/network.dt)
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	tvec = np.arange(network.tstop/network.dt+1)*network.dt
	spikebinvec = np.zeros(len(tvec))
	spikebinvec_PN = np.zeros(len(tvec))
	spikebinvec_MN = np.zeros(len(tvec))
	spikebinvec_BN = np.zeros(len(tvec))
	spikebinvec_VN = np.zeros(len(tvec))
	for i, pop in enumerate(network.populations):
		for j in range(len(SPIKES['times'][i])):
			tv = np.around(np.array(SPIKES['times'][i][j][(SPIKES['times'][i][j]>transient) & (SPIKES['times'][i][j]<stimtime)]),3)
			for x in tv: spikebinvec[np.where(tvec == x)] = spikebinvec[np.where(tvec == x)] + 1
			if i==0:
				for x in tv: spikebinvec_PN[np.where(tvec == x)] = spikebinvec_PN[np.where(tvec == x)] + 1
			if i==1:
				for x in tv: spikebinvec_MN[np.where(tvec == x)] = spikebinvec_MN[np.where(tvec == x)] + 1
			if i==2:
				for x in tv: spikebinvec_BN[np.where(tvec == x)] = spikebinvec_BN[np.where(tvec == x)] + 1
			if i==3:
				for x in tv: spikebinvec_VN[np.where(tvec == x)] = spikebinvec_VN[np.where(tvec == x)] + 1
	
	f_All, Pxx_den_All = ss.welch(spikebinvec[t1:t2], fs=sampling_rate, nperseg=nperseg)
	f_PN, Pxx_den_PN = ss.welch(spikebinvec_PN[t1:t2], fs=sampling_rate, nperseg=nperseg)
	f_MN, Pxx_den_MN = ss.welch(spikebinvec_MN[t1:t2], fs=sampling_rate, nperseg=nperseg)
	f_BN, Pxx_den_BN = ss.welch(spikebinvec_BN[t1:t2], fs=sampling_rate, nperseg=nperseg)
	f_VN, Pxx_den_VN = ss.welch(spikebinvec_VN[t1:t2], fs=sampling_rate, nperseg=nperseg)
	maxfreq_All = f_All[Pxx_den_All == np.amax(Pxx_den_All[f_All<100])]
	maxfreq_PN = f_PN[Pxx_den_PN == np.amax(Pxx_den_PN[f_All<100])]
	maxfreq_MN = f_MN[Pxx_den_MN == np.amax(Pxx_den_MN[f_All<100])]
	maxfreq_BN = f_BN[Pxx_den_BN == np.amax(Pxx_den_BN[f_All<100])]
	maxfreq_VN = f_VN[Pxx_den_VN == np.amax(Pxx_den_VN[f_All<100])]
	maxpow_All = Pxx_den_All[Pxx_den_All == np.amax(Pxx_den_All[f_All<100])]
	maxpow_PN = Pxx_den_PN[Pxx_den_PN == np.amax(Pxx_den_PN[f_All<100])]
	maxpow_MN = Pxx_den_MN[Pxx_den_MN == np.amax(Pxx_den_MN[f_All<100])]
	maxpow_BN = Pxx_den_BN[Pxx_den_BN == np.amax(Pxx_den_BN[f_All<100])]
	maxpow_VN = Pxx_den_VN[Pxx_den_VN == np.amax(Pxx_den_VN[f_All<100])]
	maxpows = np.array([maxpow_PN[0],maxpow_MN[0],maxpow_BN[0],maxpow_VN[0]])
	maxfreqs = np.array([maxfreq_PN[0],maxfreq_MN[0],maxfreq_BN[0],maxfreq_VN[0]])
	fig, axarr = plt.subplots(2,2,sharex=True)
	axarr[0,0].plot(f_PN, Pxx_den_PN,color=colors[0],label='PN')
	axarr[0,0].scatter(maxfreq_PN, maxpow_PN,c='k',label=str(np.around(maxfreq_PN[0],2)) + " Hz")
	axarr[0,1].plot(f_MN, Pxx_den_MN,color=colors[1],label='MN')
	axarr[0,1].scatter(maxfreq_MN, maxpow_MN,c='k',label=str(np.around(maxfreq_MN[0],2)) + " Hz")
	axarr[1,0].plot(f_BN, Pxx_den_BN,color=colors[2],label='BN')
	axarr[1,0].scatter(maxfreq_BN, maxpow_BN,c='k',label=str(np.around(maxfreq_BN[0],2)) + " Hz")
	axarr[1,1].plot(f_VN, Pxx_den_VN,color=colors[3],label='VN')
	axarr[1,1].scatter(maxfreq_VN, maxpow_VN,c='k',label=str(np.around(maxfreq_VN[0],2)) + " Hz")
	axarr[0,0].set_xlim(0,100)
	axarr[1,0].set_xlim(0,100)
	axarr[0,1].set_xlim(0,100)
	axarr[1,1].set_xlim(0,100)
	axarr[0,0].spines['right'].set_visible(False)
	axarr[1,0].spines['right'].set_visible(False)
	axarr[0,1].spines['right'].set_visible(False)
	axarr[1,1].spines['right'].set_visible(False)
	axarr[0,0].spines['top'].set_visible(False)
	axarr[1,0].spines['top'].set_visible(False)
	axarr[0,1].spines['top'].set_visible(False)
	axarr[1,1].spines['top'].set_visible(False)
	
	fig2, axarr2 = plt.subplots(1,1)
	axarr2.plot(f_All, Pxx_den_All,color=colors[0],label='All Populations')
	axarr2.scatter(maxfreq_All, maxpow_All,c='k',label=str(np.around(maxfreq_All[0],2)) + " Hz")
	axarr2.set_xlim(0,100)
	axarr2.legend()
	axarr2.set_xlabel('frequency (Hz)')
	axarr2.set_ylabel(r'$PSD (Spikes^2 / Hz)$')
	axarr2.spines['right'].set_visible(False)
	axarr2.spines['top'].set_visible(False)
	
	return fig, fig2

# Plot EEG & ECoG voltages & PSDs
def plot_eeg(network,DIPOLEMOMENT,low=.1,high=100.,order=2,stimtime=network.tstop):
	t2 = int(stimtime/0.025)
	DP = DIPOLEMOMENT['HL5PN1']
	for pop in popnames[1:]:
		DP = np.add(DP,DIPOLEMOMENT[pop])
	
	EEG = EEG_args.calc_potential(DP, L5_pos)
	EEG = EEG[0]
	
	EEG_filt = bandPassFilter(EEG[t1:t2],low,high,order)
	
	EEG_freq, EEG_ps = ss.welch(EEG_filt, fs=sampling_rate, nperseg=nperseg)
	EEGraw_freq, EEGraw_ps = ss.welch(EEG[t1:t2], fs=sampling_rate, nperseg=nperseg)
	
	tvec = np.arange((network.tstop)/(1000/sampling_rate)+1)*(1000/sampling_rate)
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	ax1.plot(tvec[t1:t2], EEG_filt, c='k')
	ax1.set_xlim(transient,stimtime)
	ax1.set_ylabel('EEG (mV)')
	ax2 = fig.add_subplot(212)
	ax2.plot(EEG_freq, EEG_ps, c='k')
	ax2.set_xlim(0,100)
	ax2.set_xlabel('Frequency (Hz)')
	
	fig2 = plt.figure(figsize=(10,10))
	ax21 = fig2.add_subplot(221)
	ax21.plot(tvec[t1:], EEG[t1:], c='k')
	ax21.set_xlim(transient,network.tstop)
	ax21.set_ylabel('EEG (mV)')
	ax22 = fig2.add_subplot(222)
	ax22.plot(EEGraw_freq, EEGraw_ps, c='k')
	ax22.set_xlim(0,100)
	ax22.set_xlabel('Frequency (Hz)')
	
	return fig, fig2

def plot_eegFFT(network,DIPOLEMOMENT,low=.1,high=100.,order=2,stimtime=network.tstop):
	t2 = int(stimtime/0.025)
	DP = DIPOLEMOMENT['HL5PN1']
	for pop in popnames[1:]:
		DP = np.add(DP,DIPOLEMOMENT[pop])
	
	EEG = EEG_args.calc_potential(DP, L5_pos)
	EEG = EEG[0]
	
	EEG_filt = bandPassFilter(EEG[t1:t2],low,high,order)
	
	tlength = (stimtime-transient)/1000
	
	yf = scipy.fftpack.fft(EEG_filt)
	EEG_ps = 2/(sampling_rate*tlength) * np.abs(yf[1:int(1+(sampling_rate*tlength)//2)])
	EEG_freq = np.linspace(0, sampling_rate/2, int(sampling_rate*tlength)//2)
	
	yf = scipy.fftpack.fft(EEG[t1:t2])
	EEGraw_ps = 2/(sampling_rate*tlength) * np.abs(yf[1:int(1+(sampling_rate*tlength)//2)])
	EEGraw_freq = np.linspace(0, sampling_rate/2, int(sampling_rate*tlength)//2)
	
	tvec = np.arange((network.tstop)/(1000/sampling_rate)+1)*(1000/sampling_rate)
	
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(311)
	ax1.plot([(x-transient)/1000 for x in tvec[t1:t2]], EEG_filt, c='k')
	ax1.set_xlim(0.5,4.45)
	ax1.set_ylabel('EEG [mV]')
	ax1.set_xlabel('Time [sec]')
	ax2 = fig.add_subplot(313)
	ax2.plot(EEG_freq, EEG_ps, c='k')
	ax2.set_xlim(0,50)
	ax2.set_xlabel('Frequency [Hz]')
	ax2.set_ylabel('Power')
	ax3 = fig.add_subplot(312)
	f, t, Sxx = ss.spectrogram(EEG_filt, sampling_rate, nperseg=4000)
	ax3.pcolormesh(t, f, Sxx, shading='gouraud')
	ax3.set_xlim(0.5,4.45)
	ax3.set_ylim(0,80)
	ax3.set_ylabel('Frequency [Hz]')
	ax3.set_xlabel('Time [sec]')
	fig.tight_layout()
	
	fig2 = plt.figure(figsize=(10,10))
	ax12 = fig2.add_subplot(311)
	ax12.plot([(x-transient)/1000 for x in tvec[t1:]], EEG[t1:], c='k')
	ax12.set_xlim(0.5,4.95)
	ax12.set_ylabel('EEG [mV]')
	ax12.set_xlabel('Time [sec]')
	ax22 = fig2.add_subplot(313)
	ax22.plot(EEGraw_freq, EEGraw_ps, c='k')
	ax22.set_xlim(0,50)
	ax22.set_xlabel('Frequency [Hz]')
	ax22.set_ylabel('Power')
	ax32 = fig2.add_subplot(312)
	f, t, Sxx = ss.spectrogram(EEG[t1:], sampling_rate, nperseg=4000)
	ax32.pcolormesh(t, f, Sxx, shading='gouraud')
	ax32.set_xlim(0.5,4.95)
	ax32.set_ylim(0,80)
	ax32.set_ylabel('Frequency [Hz]')
	ax32.set_xlabel('Time [sec]')
	fig2.tight_layout()
	
	return fig, fig2

# Plot LFP voltages & PSDs
def plot_lfp(network,OUTPUT,stimtime=network.tstop):
	t2 = int(stimtime/0.025)
	LFP1_freq, LFP1_ps = ss.welch(OUTPUT[0]['imem'][0][t1:t2], fs=sampling_rate, nperseg=nperseg)
	
	tvec = np.arange((network.tstop)/(1000/sampling_rate)+1)*(1000/sampling_rate)
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	ax1.plot(tvec[t1:],OUTPUT[0]['imem'][0][t1:],'k')
	ax1.set_xlim(transient,network.tstop)
	ax1.set_xlabel('Time (ms)')
	
	ax2 = fig.add_subplot(212)
	ax2.plot(LFP1_freq,LFP1_ps,'k')
	ax2.set_xlim(0,100)
	ax2.set_xlabel('Frequency (Hz)')
	fig.tight_layout()
	
	return fig

def plot_lfpFFT(network,OUTPUT,stimtime=network.tstop):
	t2 = int(stimtime/0.025)
	tlength = (stimtime-transient)/1000
	
	yf1 = scipy.fftpack.fft(OUTPUT[0]['imem'][0][t1:t2])
	LFP1_ps = 2/(sampling_rate*tlength) * np.abs(yf1[1:int(1+(sampling_rate*tlength)//2)])
	LFP1_freq = np.linspace(0, sampling_rate/2, int(sampling_rate*tlength)//2)
	
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	ax1.plot(LFP1_freq,LFP1_ps,'k')
	ax1.set_xlim(0,100)
	ax1.set_xlabel('Frequency (Hz)')
	
	ax2 = fig.add_subplot(212)
	f, t, Sxx = ss.spectrogram(OUTPUT[0]['imem'][0][t1:], sampling_rate, nperseg=4000)
	ax2.pcolormesh(t, f, Sxx, shading='gouraud')
	ax2.set_xlim(0.5,4.95)
	ax2.set_ylim(0,80)
	ax2.set_xlabel('Time (s)')
	fig.tight_layout()
	
	return fig

# Collect Somatic Voltages Across Ranks
def somavCollect(network,cellindices,RANK,SIZE,COMM):
	if RANK == 0:
		volts = []
		gids2 = []
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)
			
			volts.append([])
			gids2.append([])
			volts[i] += svolts
			gids2[i] += sgids
			
			for j in range(1, SIZE):
				volts[i] += COMM.recv(source=j, tag=15)
				gids2[i] += COMM.recv(source=j, tag=16)
	else:
		volts = None
		gids2 = None
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)
			COMM.send(svolts, dest=0, tag=15)
			COMM.send(sgids, dest=0, tag=16)
	
	return dict(volts=volts, gids2=gids2)

# Plot somatic voltages for each population
def plot_somavs(network,VOLTAGES,gstart=transient,gstop=network.tstop,tstim=0):
	tvec = np.arange(network.tstop/network.dt+1)*network.dt-tstim
	fig = plt.figure(figsize=(10,5))
	cls = ['black','crimson','green','darkorange']
	for i, pop in enumerate(network.populations):
		for v in range(0,len(VOLTAGES['volts'][i])):
			ax = plt.subplot2grid((len(VOLTAGES['volts']), len(VOLTAGES['volts'][i])), (i, v), rowspan=1, colspan=1, frameon=False)
			ax.plot(tvec,VOLTAGES['volts'][i][v], c=cls[i])
			ax.set_xlim(gstart,gstop)
			ax.set_ylim(-85,45)
			if i < len(VOLTAGES['volts'])-1:
				ax.set_xticks([])
			if v > 0:
				ax.set_yticks([])
	
	return fig

def plot_syns(network,cellindices):
	for name, pop in network.populations.items():
		for gid_cell, cell in zip(pop.gids, pop.cells):
			if gid_cell in cellindices:
				fig = plt.figure(figsize=[10, 15])
				ax = fig.add_subplot(111,frameon=False)
				for i, idx in enumerate(cell.synidx):
					if cell.netconsynapses[i].e == -80: # if inhibitory
						ax.plot(cell.ymid[idx], cell.zmid[idx], c='red', marker='.', markersize='15', alpha=0.3)
					elif cell.netconsynapses[i].e == 0: # if excitatory
						ax.plot(cell.ymid[idx], cell.zmid[idx], c='blue', marker='.', markersize='15', alpha=0.3)
				zips = []
				xlist = []
				zlist = []
				for x, z in cell.get_pt3d_polygons(projection=('y', 'z')):
					zips.append(list(zip(x, z)))
					xlist.append(x)
					zlist.append(z)
					polycol = PolyCollection(zips,
										edgecolors='none',
										facecolors='gray')
					ax.add_collection(polycol)
					# ax.set_xticks([])
					# ax.set_yticks([])
				ax.set_xlim(np.min([np.min(x1) for x1 in xlist])-5,np.max([np.max(x1) for x1 in xlist])+5)
				ax.set_ylim(np.min([np.min(z1) for z1 in zlist])-5,np.max([np.max(z1) for z1 in zlist])+5)
				fig.savefig(os.path.join(OUTPUTPATH,'synlocs_cell'+str(gid_cell)+'_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300)

# Run Plot Functions
if plotsomavs:
	cell_indices_to_plot = [0, N_HL5PN, N_HL5PN+N_HL5MN, N_HL5PN+N_HL5MN+N_HL5BN] # Plot first cell from each population
	# cell_indices_to_plot = [0, 1, 5, 801, 802, 851, 960, 961, 962, 963] # Or just choose them manually
	VOLTAGES = somavCollect(network,cell_indices_to_plot,RANK,SIZE,COMM)

if plotsynlocs:
	cell_indices_to_plot2 = [100, 101, 102, 103, N_HL5PN, N_HL5PN+1, N_HL5PN+2, N_HL5PN+3, N_HL5PN+N_HL5MN+85, N_HL5PN+N_HL5MN+85+1, N_HL5PN+N_HL5MN+85+2, N_HL5PN+N_HL5MN+85+3, N_HL5PN+N_HL5MN+N_HL5BN+26, N_HL5PN+N_HL5MN+N_HL5BN+26+1, N_HL5PN+N_HL5MN+N_HL5BN+26+2, N_HL5PN+N_HL5MN+N_HL5BN+26+3] # Plot first cell from each population
	plot_syns(network,cell_indices_to_plot2)

if RANK ==0:
	if plotnetworksomas:
		fig = plot_network_somas(OUTPUTPATH)
		fig.savefig(os.path.join(OUTPUTPATH,'network_somas_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotrasterandrates:
		fig, fig2, fig3, fig4, fig5 = plot_raster_and_rates(SPIKES,3700,5300,popnames,N_cells,network,OUTPUTPATH,GLOBALSEED,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'raster_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'rates_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig3.savefig(os.path.join(OUTPUTPATH,'rates_silentExcluded_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig4.savefig(os.path.join(OUTPUTPATH,'rates_stimchange_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig5.savefig(os.path.join(OUTPUTPATH,'rates_silentExcluded_stimchange_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig, fig2, fig3, fig4, fig5 = plot_raster_and_rates(SPIKES,6200,6800,popnames,N_cells,network,OUTPUTPATH,GLOBALSEED,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'rasterStim_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_spiketimehists(SPIKES,network)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimes_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_spiketimehists(SPIKES,network,gstart=-200,gstop=200,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimesStim_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_spiketimehists(SPIKES,network,gstart=-100,gstop=100,stimtime=6500,binsize=3)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimesStim3msBins_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotephistimseriesandPSD:
		fig, fig2 = plot_eeg(network,DIPOLEMOMENT,.1,100.,2,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'eeg_filt_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'eeg_raw_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig, fig2 = plot_eegFFT(network,DIPOLEMOMENT,.1,100.,2,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'eeg_filtFFT_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'eeg_rawFFT_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_lfp(network,OUTPUT,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'lfps_traces_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_lfpFFT(network,OUTPUT,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'lfps_PSDsFFT_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig, fig2 = plot_spikevecPSDs(SPIKES,network,stimtime=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'spikePops_PSDs_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'spikeAll_PSDs_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotsomavs:
		fig = plot_somavs(network,VOLTAGES)
		fig.savefig(os.path.join(OUTPUTPATH,'somav_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_somavs(network,VOLTAGES,gstart=-200,gstop=200,tstim=6500)
		fig.savefig(os.path.join(OUTPUTPATH,'somavStim_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
