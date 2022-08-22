################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_style("white")
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import LFPy

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

N_seeds = 30
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
dt = 0.025
tstop = 7000
startsclice = 2000
endsclice = 6500
stimtime = 6500
t1 = int(startsclice*(1/dt))
t2 = int(endsclice*(1/dt))
tstim = int(stimtime*(1/dt))
tvec = np.arange(tstop/dt+1)*dt
fs = 16000

radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L5_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

manipulation = 'Old'
condition = 'dend'
normal = '1_y_'+condition+'_0'
condit = '1_o_'+condition+'_0'
normalpath = normal + '/HL5netLFPy/Circuit_output/'
conditpath = condit + '/HL5netLFPy/Circuit_output/'

colors = ['dimgrey', 'red', 'green', 'orange']

def autoscale_y(ax,margin=0.1,multiplier=1):
	"""This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
	ax -- a matplotlib axes object
	margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

	import numpy as np

	def get_bottom_top(line):
		xd = line.get_xdata()
		yd = line.get_ydata()
		lo,hi = ax.get_xlim()
		y_displayed = yd[((xd>lo) & (xd<hi))]
		h = np.max(y_displayed) - np.min(y_displayed)
		bot = np.min(y_displayed)-margin*h
		top = np.max(y_displayed)+margin*h*multiplier
		return bot,top

	lines = ax.get_lines()
	bot,top = np.inf, -np.inf

	for line in lines[:-1]:
		new_bot, new_top = get_bottom_top(line)
		if new_bot < bot: bot = new_bot
		if new_top > top: top = new_top
	
	ax.set_ylim(bot,top)

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
	"""
	Annotate barplot with p-values.

	:param num1: number of left bar to put bracket over
	:param num2: number of right bar to put bracket over
	:param data: string to write or number for generating asterixes
	:param center: centers of all bars (like plt.bar() input)
	:param height: heights of all bars (like plt.bar() input)
	:param yerr: yerrs of all bars (like plt.bar() input)
	:param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
	:param barh: bar height in axes coordinates (0 to 1)
	:param fs: font size
	:param maxasterix: maximum number of asterixes to write (for very small p-values)
	"""

	if type(data) is str:
		text = data
	else:
		# * is p < 0.05
		# ** is p < 0.005
		# *** is p < 0.0005
		# etc.
		text = ''
		p = .05

		while data < p:
			text += '*'
			p /= 10.

			if maxasterix and len(text) == maxasterix:
				break

		if len(text) == 0:
			text = 'n. s.'

	lx, ly = center[num1], height[num1]
	rx, ry = center[num2], height[num2]

	if yerr:
		ly += yerr[num1]
		ry += yerr[num2]

	ax_y0, ax_y1 = plt.gca().get_ylim()
	dh *= (ax_y1 - ax_y0)
	barh *= (ax_y1 - ax_y0)

	y = max(ly, ry) + dh

	barx = [lx, lx, rx, rx]
	bary = [y, y+barh/2, y+barh/2, y]
	mid = ((lx+rx)/2, y+barh*(3/4))

	plt.plot(barx, bary, c='black')

	kwargs = dict(ha='center', va='bottom')
	if fs is not None:
		kwargs['fontsize'] = fs

	plt.text(*mid, text, **kwargs)

normalrates = []
conditrates = []
normalratesSE = []
conditratesSE = []
normalpercentsilent = []
conditpercentsilent = []

normaleeg = []
conditeeg = []

normallfp1 = []
conditlfp1 = []

normalspikes = []
normalspikes_PN = []
normalspikes_MN = []
normalspikes_BN = []
normalspikes_VN = []

conditspikes = []
conditspikes_PN = []
conditspikes_MN = []
conditspikes_BN = []
conditspikes_VN = []

for i in N_seedsList:
	print('Testing Seed #'+str(i))
	
	temp_sn = np.load(normalpath + 'SPIKES_Seed' + str(i) + '.npy',allow_pickle=True)
	temp_cn = np.load(conditpath + 'SPIKES_Seed' + str(i) + '.npy',allow_pickle=True)
	
	SPIKES = temp_sn.item()
	temp_rn = []
	temp_rnSE = []
	temp_rnS = []
	PN = np.zeros(len(SPIKES['times'][0]))
	MN = np.zeros(len(SPIKES['times'][1]))
	BN = np.zeros(len(SPIKES['times'][2]))
	VN = np.zeros(len(SPIKES['times'][3]))
	SPIKE_list = [PN ,MN, BN, VN]
	SILENT_list = np.zeros(len(SPIKE_list))
	for i in range(0,len(SPIKES['times'])):
		for j in range(0,len(SPIKES['times'][i])):
			scount = SPIKES['times'][i][j][(SPIKES['times'][i][j]>startsclice) & (SPIKES['times'][i][j]<=stimtime)]
			scount2 = SPIKES['times'][i][j][(SPIKES['times'][i][j]>(stimtime+5)) & (SPIKES['times'][i][j]<=(stimtime+55))]
			Hz = (scount.size)/((int(stimtime)-startsclice)/1000)
			Hz2 = (scount2.size)/((int(stimtime+55)-int(stimtime+5))/1000)
			SPIKE_list[i][j] = Hz
			if Hz <= 0.2:
				SILENT_list[i] += 1
		temp_rn.append(np.mean(SPIKE_list[i]))
		temp_rnSE.append(np.mean(SPIKE_list[i][SPIKE_list[i]>0.2]))
		temp_rnS.append((SILENT_list[i]/len(SPIKES['times'][i]))*100)
	
	SPIKES = temp_cn.item()
	temp_rc = []
	temp_rcSE = []
	temp_rcS = []
	PN = np.zeros(len(SPIKES['times'][0]))
	MN = np.zeros(len(SPIKES['times'][1]))
	BN = np.zeros(len(SPIKES['times'][2]))
	VN = np.zeros(len(SPIKES['times'][3]))
	SPIKE_list = [PN ,MN, BN, VN]
	SILENT_list = np.zeros(len(SPIKE_list))
	for i in range(0,len(SPIKES['times'])):
		for j in range(0,len(SPIKES['times'][i])):
			scount = SPIKES['times'][i][j][(SPIKES['times'][i][j]>startsclice) & (SPIKES['times'][i][j]<=stimtime)]
			scount2 = SPIKES['times'][i][j][(SPIKES['times'][i][j]>(stimtime+5)) & (SPIKES['times'][i][j]<=(stimtime+55))]
			Hz = (scount.size)/((int(stimtime)-startsclice)/1000)
			Hz2 = (scount2.size)/((int(stimtime+55)-int(stimtime+5))/1000)
			SPIKE_list[i][j] = Hz
			if Hz <= 0.2:
				SILENT_list[i] += 1
		temp_rc.append(np.mean(SPIKE_list[i]))
		temp_rcSE.append(np.mean(SPIKE_list[i][SPIKE_list[i]>0.2]))
		temp_rcS.append((SILENT_list[i]/len(SPIKES['times'][i]))*100)
	
	SPIKES1 = [x for _,x in sorted(zip(temp_sn.item()['gids'][0],temp_sn.item()['times'][0]))]
	SPIKES2 = [x for _,x in sorted(zip(temp_sn.item()['gids'][1],temp_sn.item()['times'][1]))]
	SPIKES3 = [x for _,x in sorted(zip(temp_sn.item()['gids'][2],temp_sn.item()['times'][2]))]
	SPIKES4 = [x for _,x in sorted(zip(temp_sn.item()['gids'][3],temp_sn.item()['times'][3]))]
	SPIKES_all = SPIKES1+SPIKES2+SPIKES3+SPIKES4
	
	popspikes_All = np.concatenate(SPIKES_all).ravel()
	popspikes_PN = np.concatenate(SPIKES1).ravel()
	popspikes_MN = np.concatenate(SPIKES2).ravel()
	popspikes_BN = np.concatenate(SPIKES3).ravel()
	popspikes_VN = np.concatenate(SPIKES4).ravel()
	spikebinvec = np.histogram(popspikes_All,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_PN = np.histogram(popspikes_PN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_MN = np.histogram(popspikes_MN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_BN = np.histogram(popspikes_BN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_VN = np.histogram(popspikes_VN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	
	sampling_rate = (1/dt)*1000
	nperseg = len(tvec[t1:tstim])/2
	f_All, Pxx_den_All = ss.welch(spikebinvec, fs=sampling_rate, nperseg=nperseg)
	f_PN, Pxx_den_PN = ss.welch(spikebinvec_PN, fs=sampling_rate, nperseg=nperseg)
	f_MN, Pxx_den_MN = ss.welch(spikebinvec_MN, fs=sampling_rate, nperseg=nperseg)
	f_BN, Pxx_den_BN = ss.welch(spikebinvec_BN, fs=sampling_rate, nperseg=nperseg)
	f_VN, Pxx_den_VN = ss.welch(spikebinvec_VN, fs=sampling_rate, nperseg=nperseg)
	normalspikes.append(Pxx_den_All)
	normalspikes_PN.append(Pxx_den_PN)
	normalspikes_MN.append(Pxx_den_MN)
	normalspikes_BN.append(Pxx_den_BN)
	normalspikes_VN.append(Pxx_den_VN)
	
	SPIKES1 = [x for _,x in sorted(zip(temp_cn.item()['gids'][0],temp_cn.item()['times'][0]))]
	SPIKES2 = [x for _,x in sorted(zip(temp_cn.item()['gids'][1],temp_cn.item()['times'][1]))]
	SPIKES3 = [x for _,x in sorted(zip(temp_cn.item()['gids'][2],temp_cn.item()['times'][2]))]
	SPIKES4 = [x for _,x in sorted(zip(temp_cn.item()['gids'][3],temp_cn.item()['times'][3]))]
	SPIKES_all = SPIKES1+SPIKES2+SPIKES3+SPIKES4
	
	popspikes_All = np.concatenate(SPIKES_all).ravel()
	popspikes_PN = np.concatenate(SPIKES1).ravel()
	popspikes_MN = np.concatenate(SPIKES2).ravel()
	popspikes_BN = np.concatenate(SPIKES3).ravel()
	popspikes_VN = np.concatenate(SPIKES4).ravel()
	spikebinvec = np.histogram(popspikes_All,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_PN = np.histogram(popspikes_PN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_MN = np.histogram(popspikes_MN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_BN = np.histogram(popspikes_BN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	spikebinvec_VN = np.histogram(popspikes_VN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
	
	sampling_rate = (1/dt)*1000
	nperseg = len(tvec[t1:tstim])/2
	f_All, Pxx_den_All = ss.welch(spikebinvec, fs=sampling_rate, nperseg=nperseg)
	f_PN, Pxx_den_PN = ss.welch(spikebinvec_PN, fs=sampling_rate, nperseg=nperseg)
	f_MN, Pxx_den_MN = ss.welch(spikebinvec_MN, fs=sampling_rate, nperseg=nperseg)
	f_BN, Pxx_den_BN = ss.welch(spikebinvec_BN, fs=sampling_rate, nperseg=nperseg)
	f_VN, Pxx_den_VN = ss.welch(spikebinvec_VN, fs=sampling_rate, nperseg=nperseg)
	conditspikes.append(Pxx_den_All)
	conditspikes_PN.append(Pxx_den_PN)
	conditspikes_MN.append(Pxx_den_MN)
	conditspikes_BN.append(Pxx_den_BN)
	conditspikes_VN.append(Pxx_den_VN)
	
	# temp_rn = np.loadtxt(normalpath + 'spikerates_Seed' + str(i) + '.txt')
	# temp_rc = np.loadtxt(conditpath + 'spikerates_Seed' + str(i) + '.txt')
	# temp_rnSE = np.loadtxt(normalpath + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	# temp_rcSE = np.loadtxt(conditpath + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	# temp_rnS = np.loadtxt(normalpath + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	# temp_rcS = np.loadtxt(conditpath + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	temp_en = np.load(normalpath + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ec = np.load(conditpath + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ln = np.load(normalpath + 'OUTPUT_Seed' + str(i) + '.npy')
	temp_lc = np.load(conditpath + 'OUTPUT_Seed' + str(i) + '.npy')
	
	temp_en2 = temp_en['HL5PN1']
	temp_en2 = np.add(temp_en2,temp_en['HL5MN1'])
	temp_en2 = np.add(temp_en2,temp_en['HL5BN1'])
	temp_en2 = np.add(temp_en2,temp_en['HL5VN1'])
	
	temp_ec2 = temp_ec['HL5PN1']
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5MN1'])
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5BN1'])
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5VN1'])
	
	potentialn = EEG_args.calc_potential(temp_en2, L5_pos)
	EEGn = potentialn[0][t1:t2]
	potentialc = EEG_args.calc_potential(temp_ec2, L5_pos)
	EEGc = potentialc[0][t1:t2]
	
	nperseg = len(tvec[t1:t2])/2
	freqrawEEGn, psrawEEGn = ss.welch(EEGn, 1/(dt/1000), nperseg=nperseg)
	freqrawEEGc, psrawEEGc = ss.welch(EEGc, 1/(dt/1000), nperseg=nperseg)
	psrawLFPn = []
	psrawLFPc = []
	for l in range(0,len(temp_ln[0])):
		freqrawLFPn, psdn = ss.welch(temp_ln[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=nperseg)
		freqrawLFPc, psdc = ss.welch(temp_lc[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=nperseg)
		psrawLFPn.append(psdn)
		psrawLFPc.append(psdc)
	
	normalrates.append(temp_rn)
	conditrates.append(temp_rc)
	normalratesSE.append(temp_rnSE)
	conditratesSE.append(temp_rcSE)
	normalpercentsilent.append(temp_rnS)
	conditpercentsilent.append(temp_rcS)
	
	normaleeg.append(psrawEEGn)
	conditeeg.append(psrawEEGc)
	
	normallfp1.append(psrawLFPn[0])
	conditlfp1.append(psrawLFPc[0])

meanSpikePSDn = np.mean(normalspikes,0)
meanSpikePSDn_PN = np.mean(normalspikes_PN,0)
meanSpikePSDn_MN = np.mean(normalspikes_MN,0)
meanSpikePSDn_BN = np.mean(normalspikes_BN,0)
meanSpikePSDn_VN = np.mean(normalspikes_VN,0)
stdevSpikePSDn = np.std(normalspikes,0)
stdevSpikePSDn_PN = np.std(normalspikes_PN,0)
stdevSpikePSDn_MN = np.std(normalspikes_MN,0)
stdevSpikePSDn_BN = np.std(normalspikes_BN,0)
stdevSpikePSDn_VN = np.std(normalspikes_VN,0)

freq0 = 1
freq1 = 40
freq2 = 100
f1 = np.where(f_All>=freq0)
f1 = f1[0][0]
f2 = np.where(f_All>=freq1)
f2 = f2[0][0]
f3 = np.where(f_All>=freq2)
f3 = f3[0][0]

# Compute 95% confidence interval of spike PSD for PNs
CI_means_PN = []
CI_lower_PN = []
CI_upper_PN = []
CI_means_PNc = []
CI_lower_PNc = []
CI_upper_PNc = []
print('Starting Bootstrapping')
for l in range(0,len(normalspikes_PN[0][:f3])):
	x = bs.bootstrap(np.transpose(normalspikes_PN)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	CI_means_PN.append(x.value)
	CI_lower_PN.append(x.lower_bound)
	CI_upper_PN.append(x.upper_bound)
	x = bs.bootstrap(np.transpose(conditspikes_PN)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	CI_means_PNc.append(x.value)
	CI_lower_PNc.append(x.lower_bound)
	CI_upper_PNc.append(x.upper_bound)
print('Bootstrapping Done')

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(f_All[f1:f2], meanSpikePSDn[f1:f2], 'k')
ax.fill_between(f_All[f1:f2], meanSpikePSDn[f1:f2]-stdevSpikePSDn[f1:f2], meanSpikePSDn[f1:f2]+stdevSpikePSDn[f1:f2],color='k',alpha=0.4)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(f_All[f1:f3], meanSpikePSDn[f1:f3], 'k')
inset.set_xscale('log')
inset.set_yscale('log')
inset.tick_params(axis='x', which='major', bottom=True)
inset.tick_params(axis='y', which='major', left=True)
inset.tick_params(axis='x', which='minor', bottom=True)
inset.tick_params(axis='y', which='minor', left=True)
inset.xaxis.set_minor_locator(MultipleLocator(5))
# autoscale_y(inset,margin=1e-3,multiplier=3000)
inset.fill_between(f_All[f1:f3], meanSpikePSDn[f1:f3]-stdevSpikePSDn[f1:f3], meanSpikePSDn[f1:f3]+stdevSpikePSDn[f1:f3],color='k',alpha=0.4)
inset.set_xlim(freq0,freq2)
# ylims = inset.get_ylim()
# inset.set_ylim(ylims)
ax.set_xlim(freq0,freq1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Spikes PSD')
fig.savefig('figs/Spikes_PSD.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(f_PN[f1:f2], meanSpikePSDn_PN[f1:f2], 'k')
ax.fill_between(f_PN[f1:f2], meanSpikePSDn_PN[f1:f2]-stdevSpikePSDn_PN[f1:f2], meanSpikePSDn_PN[f1:f2]+stdevSpikePSDn_PN[f1:f2],color='k',alpha=0.4)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(f_PN[f1:f3], meanSpikePSDn_PN[f1:f3], 'k')
inset.set_xscale('log')
inset.set_yscale('log')
inset.tick_params(axis='x', which='major', bottom=True)
inset.tick_params(axis='y', which='major', left=True)
inset.tick_params(axis='x', which='minor', bottom=True)
inset.tick_params(axis='y', which='minor', left=True)
inset.xaxis.set_minor_locator(MultipleLocator(5))
# autoscale_y(inset,margin=1e-3,multiplier=3000)
inset.fill_between(f_PN[f1:f3], meanSpikePSDn_PN[f1:f3]-stdevSpikePSDn_PN[f1:f3], meanSpikePSDn_PN[f1:f3]+stdevSpikePSDn_PN[f1:f3],color='k',alpha=0.4)
inset.set_xlim(freq0,freq2)
# ylims = inset.get_ylim()
# inset.set_ylim(ylims)
ax.set_xlim(freq0,freq1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Spikes PSD')
fig.savefig('figs/Spikes_PSD_PN.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(f_PN[f1:f2], CI_means_PN[f1:f2], 'k')
ax.fill_between(f_PN[f1:f2], CI_lower_PN[f1:f2], CI_upper_PN[f1:f2],color='k',alpha=0.4)
ax.plot(f_PN[f1:f2], CI_means_PNc[f1:f2], 'r')
ax.fill_between(f_PN[f1:f2], CI_lower_PNc[f1:f2], CI_upper_PNc[f1:f2],color='r',alpha=0.4)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(f_PN[:f3], CI_means_PN[:f3], 'k')
inset.plot(f_PN[:f3], CI_means_PNc[:f3], 'r')
inset.set_xscale('log')
inset.set_yscale('log')
inset.tick_params(axis='x', which='major', bottom=True)
inset.tick_params(axis='y', which='major', left=True)
inset.tick_params(axis='x', which='minor', bottom=True)
inset.tick_params(axis='y', which='minor', left=True)
inset.xaxis.set_minor_locator(MultipleLocator(5))
# autoscale_y(inset,margin=1e-3,multiplier=3000)
inset.fill_between(f_PN[:f3], CI_lower_PN[:f3], CI_upper_PN[:f3],color='k',alpha=0.4)
inset.fill_between(f_PN[:f3], CI_lower_PNc[:f3], CI_upper_PNc[:f3],color='r',alpha=0.4)
inset.set_xticks([1,10,freq2])
inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
inset.set_xlim(0.3,freq2)
ylims = ax.get_ylim()
ax.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
ax.set_xticks([freq0,10,20,30,freq1])
ax.set_xticklabels([str(freq0),'10','20','30',str(freq1)])
ax.set_ylim(0,ylims[1])
ax.set_xlim(freq0,freq1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Pyr Network PSD (Spikes'+r'$^{2}$'+'/Hz)')
fig.savefig('figs/Spikes_PSD_Boot95CI_PN.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

fig, axarr = plt.subplots(2,2,sharex=True)
axarr[0,0].plot(f_PN[f1:f2], meanSpikePSDn_PN[f1:f2],color=colors[0],label='PN')
axarr[0,0].fill_between(f_PN[f1:f2], meanSpikePSDn_PN[f1:f2]-stdevSpikePSDn_PN[f1:f2], meanSpikePSDn_PN[f1:f2]+stdevSpikePSDn_PN[f1:f2],color=colors[0],alpha=0.4)
axarr[0,1].plot(f_MN[f1:f2], meanSpikePSDn_MN[f1:f2],color=colors[1],label='MN')
axarr[0,1].fill_between(f_MN[f1:f2], meanSpikePSDn_MN[f1:f2]-stdevSpikePSDn_MN[f1:f2], meanSpikePSDn_MN[f1:f2]+stdevSpikePSDn_MN[f1:f2],color=colors[1],alpha=0.4)
axarr[1,0].plot(f_BN[f1:f2], meanSpikePSDn_BN[f1:f2],color=colors[2],label='BN')
axarr[1,0].fill_between(f_BN[f1:f2], meanSpikePSDn_BN[f1:f2]-stdevSpikePSDn_BN[f1:f2], meanSpikePSDn_BN[f1:f2]+stdevSpikePSDn_BN[f1:f2],color=colors[2],alpha=0.4)
axarr[1,1].plot(f_VN[f1:f2], meanSpikePSDn_VN[f1:f2],color=colors[3],label='VN')
axarr[1,1].fill_between(f_VN[f1:f2], meanSpikePSDn_VN[f1:f2]-stdevSpikePSDn_VN[f1:f2], meanSpikePSDn_VN[f1:f2]+stdevSpikePSDn_VN[f1:f2],color=colors[3],alpha=0.4)
axarr[0,0].set_xlim(freq0,freq1)
axarr[1,0].set_xlim(freq0,freq1)
axarr[0,1].set_xlim(freq0,freq1)
axarr[1,1].set_xlim(freq0,freq1)
axarr[0,0].spines['right'].set_visible(False)
axarr[1,0].spines['right'].set_visible(False)
axarr[0,1].spines['right'].set_visible(False)
axarr[1,1].spines['right'].set_visible(False)
axarr[0,0].spines['top'].set_visible(False)
axarr[1,0].spines['top'].set_visible(False)
axarr[0,1].spines['top'].set_visible(False)
axarr[1,1].spines['top'].set_visible(False)
fig.savefig('figs/Spikes_Types_PSD.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

meanRatesn = np.mean(normalrates,0)
meanRatesc = np.mean(conditrates,0)
meanRatesnSE = np.mean(normalratesSE,0)
meanRatescSE = np.mean(conditratesSE,0)
meanSilentnS = np.mean(normalpercentsilent,0)
meanSilentcS = np.mean(conditpercentsilent,0)

stdevRatesn = np.std(normalrates,0)
stdevRatesc = np.std(conditrates,0)
stdevRatesnSE = np.std(normalratesSE,0)
stdevRatescSE = np.std(conditratesSE,0)
stdevSilentnS = np.std(normalpercentsilent,0)
stdevSilentcS = np.std(conditpercentsilent,0)

meanRates = [meanRatesn[0],meanRatesc[0],meanRatesn[1],meanRatesc[1],meanRatesn[2],meanRatesc[2],meanRatesn[3],meanRatesc[3]]
meanRatesSE = [meanRatesnSE[0],meanRatescSE[0],meanRatesnSE[1],meanRatescSE[1],meanRatesnSE[2],meanRatescSE[2],meanRatesnSE[3],meanRatescSE[3]]
meanSilent = [meanSilentnS[0],meanSilentcS[0],meanSilentnS[1],meanSilentcS[1],meanSilentnS[2],meanSilentcS[2],meanSilentnS[3],meanSilentcS[3]]

stdevRate = [stdevRatesn[0],stdevRatesc[0],stdevRatesn[1],stdevRatesc[1],stdevRatesn[2],stdevRatesc[2],stdevRatesn[3],stdevRatesc[3]]
stdevRateSE = [stdevRatesnSE[0],stdevRatescSE[0],stdevRatesnSE[1],stdevRatescSE[1],stdevRatesnSE[2],stdevRatescSE[2],stdevRatesnSE[3],stdevRatescSE[3]]
stdevSilent = [stdevSilentnS[0],stdevSilentcS[0],stdevSilentnS[1],stdevSilentcS[1],stdevSilentnS[2],stdevSilentcS[2],stdevSilentnS[3],stdevSilentcS[3]]

nr = np.array(normalrates).transpose()
nc = np.array(conditrates).transpose()
Rates = [nr[0],nc[0],nr[1],nc[1],nr[2],nc[2],nr[3],nc[3]]

nrSE = np.array(normalratesSE).transpose()
ncSE = np.array(conditratesSE).transpose()
RatesSE = [nrSE[0],ncSE[0],nrSE[1],ncSE[1],nrSE[2],ncSE[2],nrSE[3],ncSE[3]]

nrS = np.array(normalpercentsilent).transpose()
ncS = np.array(conditpercentsilent).transpose()
Silent = [nrS[0],ncS[0],nrS[1],ncS[1],nrS[2],ncS[2],nrS[3],ncS[3]]

tstat_PN, pval_PN = st.ttest_rel(nr[0],nc[0])
tstat_MN, pval_MN = st.ttest_rel(nr[1],nc[1])
tstat_BN, pval_BN = st.ttest_rel(nr[2],nc[2])
tstat_VN, pval_VN = st.ttest_rel(nr[3],nc[3])

tstat_PNSE, pval_PNSE = st.ttest_rel(nrSE[0],ncSE[0])
tstat_MNSE, pval_MNSE = st.ttest_rel(nrSE[1],ncSE[1])
tstat_BNSE, pval_BNSE = st.ttest_rel(nrSE[2],ncSE[2])
tstat_VNSE, pval_VNSE = st.ttest_rel(nrSE[3],ncSE[3])

tstat_PNS, pval_PNS = st.ttest_rel(nrS[0],ncS[0])
tstat_MNS, pval_MNS = st.ttest_rel(nrS[1],ncS[1])
tstat_BNS, pval_BNS = st.ttest_rel(nrS[2],ncS[2])
tstat_VNS, pval_VNS = st.ttest_rel(nrS[3],ncS[3])

meanRates_PNonly = [meanRatesn[0],meanRatesc[0]]
stdevRate_PNonly = [stdevRatesn[0],stdevRatesc[0]]
Rates_PNonly = [nr[0],nc[0]]

meanRates_PNonlySE = [meanRatesnSE[0],meanRatescSE[0]]
stdevRate_PNonlySE = [stdevRatesnSE[0],stdevRatescSE[0]]
Rates_PNonlySE = [nrSE[0],ncSE[0]]

meanSilent_PNonly = [meanSilentnS[0],meanSilentcS[0]]
stdevSilent_PNonly = [stdevSilentnS[0],stdevSilentcS[0]]
Silent_PNonlyS = [nrS[0],ncS[0]]

meanstdevstr1n = '\n' + str(np.around(meanRatesn[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesn[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesn[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesn[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanRatesc[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanRatesc[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanRatesc[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanRatesc[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
names = [namesn[0],namesc[0],namesn[1],namesc[1],namesn[2],namesc[2],namesn[3],namesc[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7]
colors = ['dimgray', 'dimgray', 'red', 'red', 'green', 'green', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRates,
	   yerr=stdevRate,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PN < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanRates, yerr=stdevRate)
elif pval_PN < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanRates, yerr=stdevRate)
elif pval_PN < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanRates, yerr=stdevRate)

if pval_MN < 0.001:
	barplot_annotate_brackets(2, 3, '***', x, meanRates, yerr=stdevRate)
elif pval_MN < 0.01:
	barplot_annotate_brackets(2, 3, '**', x, meanRates, yerr=stdevRate)
elif pval_MN < 0.05:
	barplot_annotate_brackets(2, 3, '*', x, meanRates, yerr=stdevRate)

if pval_BN < 0.001:
	barplot_annotate_brackets(4, 5, '***', x, meanRates, yerr=stdevRate)
elif pval_BN < 0.01:
	barplot_annotate_brackets(4, 5, '**', x, meanRates, yerr=stdevRate)
elif pval_BN < 0.05:
	barplot_annotate_brackets(4, 5, '*', x, meanRates, yerr=stdevRate)

if pval_VN < 0.001:
	barplot_annotate_brackets(6, 7, '***', x, meanRates, yerr=stdevRate)
elif pval_VN < 0.01:
	barplot_annotate_brackets(6, 7, '**', x, meanRates, yerr=stdevRate)
elif pval_VN < 0.05:
	barplot_annotate_brackets(6, 7, '*', x, meanRates, yerr=stdevRate)

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/Rates.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesnSE[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesnSE[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesnSE[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesnSE[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanRatescSE[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanRatescSE[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanRatescSE[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanRatescSE[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
names = [namesn[0],namesc[0],namesn[1],namesc[1],namesn[2],namesc[2],namesn[3],namesc[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7]
colors = ['dimgray', 'dimgray', 'red', 'red', 'green', 'green', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRatesSE,
	   yerr=stdevRateSE,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PNSE < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanRatesSE, yerr=stdevRateSE)
elif pval_PNSE < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanRatesSE, yerr=stdevRateSE)
elif pval_PNSE < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanRatesSE, yerr=stdevRateSE)

if pval_MNSE < 0.001:
	barplot_annotate_brackets(2, 3, '***', x, meanRatesSE, yerr=stdevRateSE)
elif pval_MNSE < 0.01:
	barplot_annotate_brackets(2, 3, '**', x, meanRatesSE, yerr=stdevRateSE)
elif pval_MNSE < 0.05:
	barplot_annotate_brackets(2, 3, '*', x, meanRatesSE, yerr=stdevRateSE)

if pval_BNSE < 0.001:
	barplot_annotate_brackets(4, 5, '***', x, meanRatesSE, yerr=stdevRateSE)
elif pval_BNSE < 0.01:
	barplot_annotate_brackets(4, 5, '**', x, meanRatesSE, yerr=stdevRateSE)
elif pval_BNSE < 0.05:
	barplot_annotate_brackets(4, 5, '*', x, meanRatesSE, yerr=stdevRateSE)

if pval_VNSE < 0.001:
	barplot_annotate_brackets(6, 7, '***', x, meanRatesSE, yerr=stdevRateSE)
elif pval_VNSE < 0.01:
	barplot_annotate_brackets(6, 7, '**', x, meanRatesSE, yerr=stdevRateSE)
elif pval_VNSE < 0.05:
	barplot_annotate_brackets(6, 7, '*', x, meanRatesSE, yerr=stdevRateSE)

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/RatesSE.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanSilentnS[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanSilentnS[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanSilentnS[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanSilentnS[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanSilentcS[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanSilentcS[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanSilentcS[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanSilentcS[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
names = [namesn[0],namesc[0],namesn[1],namesc[1],namesn[2],namesc[2],namesn[3],namesc[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7]
colors = ['dimgray', 'dimgray', 'red', 'red', 'green', 'green', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanSilent,
	   yerr=stdevSilent,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PNS < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanSilent, yerr=stdevSilent)
elif pval_PNS < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanSilent, yerr=stdevSilent)
elif pval_PNS < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanSilent, yerr=stdevSilent)

if pval_MNS < 0.001:
	barplot_annotate_brackets(2, 3, '***', x, meanSilent, yerr=stdevSilent)
elif pval_MNS < 0.01:
	barplot_annotate_brackets(2, 3, '**', x, meanSilent, yerr=stdevSilent)
elif pval_MNS < 0.05:
	barplot_annotate_brackets(2, 3, '*', x, meanSilent, yerr=stdevSilent)

if pval_BNS < 0.001:
	barplot_annotate_brackets(4, 5, '***', x, meanSilent, yerr=stdevSilent)
elif pval_BNS < 0.01:
	barplot_annotate_brackets(4, 5, '**', x, meanSilent, yerr=stdevSilent)
elif pval_BNS < 0.05:
	barplot_annotate_brackets(4, 5, '*', x, meanSilent, yerr=stdevSilent)

if pval_VNS < 0.001:
	barplot_annotate_brackets(6, 7, '***', x, meanSilent, yerr=stdevSilent)
elif pval_VNS < 0.01:
	barplot_annotate_brackets(6, 7, '**', x, meanSilent, yerr=stdevSilent)
elif pval_VNS < 0.05:
	barplot_annotate_brackets(6, 7, '*', x, meanSilent, yerr=stdevSilent)

ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/SilentAll.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesn[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[0], decimals=2)) + ' Hz'
meanstdevstr2n = '\n' + str(np.around(meanRatesn[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[1], decimals=2)) + ' Hz'
meanstdevstr3n = '\n' + str(np.around(meanRatesn[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[2], decimals=2)) + ' Hz'
meanstdevstr4n = '\n' + str(np.around(meanRatesn[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[3], decimals=2)) + ' Hz'
meanstdevstr1c = '\n' + str(np.around(meanRatesc[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[0], decimals=2)) + ' Hz'
meanstdevstr2c = '\n' + str(np.around(meanRatesc[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[1], decimals=2)) + ' Hz'
meanstdevstr3c = '\n' + str(np.around(meanRatesc[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[2], decimals=2)) + ' Hz'
meanstdevstr4c = '\n' + str(np.around(meanRatesc[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[3], decimals=2)) + ' Hz'
namesn = ['Young'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
names_PNonly = [namesn[0],namesc[0]]
x = [0, 1]
colors = ['dimgray', 'red']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRates_PNonly,
	   yerr=stdevRate_PNonly,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.6,    # bar width
	   tick_label=names_PNonly,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PN < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanRates_PNonly, yerr=stdevRate_PNonly)
elif pval_PN < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanRates_PNonly, yerr=stdevRate_PNonly)
elif pval_PN < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanRates_PNonly, yerr=stdevRate_PNonly)

ax.set_xlim(-0.65,1.65)
ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/Rates_PNonly.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesnSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[0], decimals=2)) + ' Hz'
meanstdevstr2n = '\n' + str(np.around(meanRatesnSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[1], decimals=2)) + ' Hz'
meanstdevstr3n = '\n' + str(np.around(meanRatesnSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[2], decimals=2)) + ' Hz'
meanstdevstr4n = '\n' + str(np.around(meanRatesnSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[3], decimals=2)) + ' Hz'
meanstdevstr1c = '\n' + str(np.around(meanRatescSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[0], decimals=2)) + ' Hz'
meanstdevstr2c = '\n' + str(np.around(meanRatescSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[1], decimals=2)) + ' Hz'
meanstdevstr3c = '\n' + str(np.around(meanRatescSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[2], decimals=2)) + ' Hz'
meanstdevstr4c = '\n' + str(np.around(meanRatescSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[3], decimals=2)) + ' Hz'
namesn = ['Young'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
names_PNonly = [namesn[0],namesc[0]]
x = [0, 1]
colors = ['dimgray', 'red']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRates_PNonlySE,
	   yerr=stdevRate_PNonlySE,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.6,    # bar width
	   tick_label=names_PNonly,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PNSE < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE)
elif pval_PNSE < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE)
elif pval_PNSE < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE)

ax.set_xlim(-0.65,1.65)
ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/Rates_PNonlySE.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanSilentnS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[0], decimals=2)) + ' %'
meanstdevstr2n = '\n' + str(np.around(meanSilentnS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[1], decimals=2)) + ' %'
meanstdevstr3n = '\n' + str(np.around(meanSilentnS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[2], decimals=2)) + ' %'
meanstdevstr4n = '\n' + str(np.around(meanSilentnS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[3], decimals=2)) + ' %'
meanstdevstr1c = '\n' + str(np.around(meanSilentcS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[0], decimals=2)) + ' %'
meanstdevstr2c = '\n' + str(np.around(meanSilentcS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[1], decimals=2)) + ' %'
meanstdevstr3c = '\n' + str(np.around(meanSilentcS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[2], decimals=2)) + ' %'
meanstdevstr4c = '\n' + str(np.around(meanSilentcS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[3], decimals=2)) + ' %'
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
names_PNonly = [namesn[0],namesc[0]]
x = [0, 1]
colors = ['dimgray', 'red']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanSilent_PNonly,
	   yerr=stdevSilent_PNonly,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.6,    # bar width
	   tick_label=names_PNonly,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

if pval_PNS < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)
elif pval_PNS < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)
elif pval_PNS < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)

ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/PercentSilent_PNonly.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesn[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesn[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesn[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesn[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[3], decimals=2))
meanstdevstr1c = '\n\n' + str(np.around(meanRatesc[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[0], decimals=2))
meanstdevstr2c = '\n\n' + str(np.around(meanRatesc[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[1], decimals=2))
meanstdevstr3c = '\n\n' + str(np.around(meanRatesc[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[2], decimals=2))
meanstdevstr4c = '\n\n' + str(np.around(meanRatesc[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]

names = [namesn[0],namesn[1],namesn[2],namesn[3]]
x = [0, 1, 2, 3]
colors = ['dimgray', 'crimson', 'green', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRatesn,
	   yerr=stdevRatesn,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=1,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/RatesNormal.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesnSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesnSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesnSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesnSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[3], decimals=2))
meanstdevstr1c = '\n\n' + str(np.around(meanRatescSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[0], decimals=2))
meanstdevstr2c = '\n\n' + str(np.around(meanRatescSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[1], decimals=2))
meanstdevstr3c = '\n\n' + str(np.around(meanRatescSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[2], decimals=2))
meanstdevstr4c = '\n\n' + str(np.around(meanRatescSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]

names = [namesn[0],namesn[1],namesn[2],namesn[3]]
x = [0, 1, 2, 3]
colors = ['dimgray', 'crimson', 'green', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRatesnSE,
	   yerr=stdevRatesnSE,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=1,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/RatesNormalSE.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanSilentnS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanSilentnS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanSilentnS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanSilentnS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[3], decimals=2))
meanstdevstr1c = '\n\n' + str(np.around(meanSilentcS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[0], decimals=2))
meanstdevstr2c = '\n\n' + str(np.around(meanSilentcS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[1], decimals=2))
meanstdevstr3c = '\n\n' + str(np.around(meanSilentcS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[2], decimals=2))
meanstdevstr4c = '\n\n' + str(np.around(meanSilentcS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = [manipulation+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]

names = [namesn[0],namesn[1],namesn[2],namesn[3]]
x = [0, 1, 2, 3]
colors = ['dimgray', 'red', 'green', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanSilentnS,
	   yerr=stdevSilentnS,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/SilentNormal.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

names = [namesn[0]]
x = [0]
colors = ['dimgray']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanSilentnS[0],
	   yerr=stdevSilentnS[0],    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )
ax.set_xlim(-0.75,0.75)
ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs/SilentNormal_PNOnly.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanEEGPSDn = np.mean(normaleeg,0)
meanEEGPSDc = np.mean(conditeeg,0)
stdevEEGPSDn = np.std(normaleeg,0)
stdevEEGPSDc = np.std(conditeeg,0)

meanLFP1PSDn = np.mean(normallfp1,0)
meanLFP1PSDc = np.mean(conditlfp1,0)
stdevLFP1PSDn = np.std(normallfp1,0)
stdevLFP1PSDc = np.std(conditlfp1,0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(freqrawEEGn, meanEEGPSDn, 'k')
ax.plot(freqrawEEGc, meanEEGPSDc, 'r')
ax.fill_between(freqrawEEGn, meanEEGPSDn-stdevEEGPSDn, meanEEGPSDn+stdevEEGPSDn,color='k',alpha=0.4)
ax.fill_between(freqrawEEGc, meanEEGPSDc-stdevEEGPSDc, meanEEGPSDc+stdevEEGPSDc,color='r',alpha=0.4)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(freqrawEEGn, meanEEGPSDn, 'k')
inset.plot(freqrawEEGc, meanEEGPSDc, 'r')
inset.set_xscale('log')
inset.set_yscale('log')
inset.tick_params(axis='x', which='major', bottom=True)
inset.tick_params(axis='y', which='major', left=True)
inset.tick_params(axis='x', which='minor', bottom=True)
inset.tick_params(axis='y', which='minor', left=True)
inset.xaxis.set_minor_locator(MultipleLocator(5))
inset.set_xlim(1,100)
autoscale_y(inset,margin=1e-3,multiplier=3000)
ylims = inset.get_ylim()
inset.fill_between(freqrawEEGn, meanEEGPSDn-stdevEEGPSDn, meanEEGPSDn+stdevEEGPSDn,color='k',alpha=0.4)
inset.fill_between(freqrawEEGc, meanEEGPSDc-stdevEEGPSDc, meanEEGPSDc+stdevEEGPSDc,color='r',alpha=0.4)
inset.set_ylim(ylims)
ax.set_xlim(0,40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('EEG PSD')
fig.savefig('figs/EEG_PSD.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(freqrawLFPn,meanLFP1PSDn,'k')
ax.plot(freqrawLFPc,meanLFP1PSDc,'r')
ax.fill_between(freqrawLFPn, meanLFP1PSDn-stdevLFP1PSDn, meanLFP1PSDn+stdevLFP1PSDn,color='k',alpha=0.4)
ax.fill_between(freqrawLFPc, meanLFP1PSDc-stdevLFP1PSDc, meanLFP1PSDc+stdevLFP1PSDc,color='r',alpha=0.4)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(freqrawLFPn, meanLFP1PSDn, 'k')
inset.plot(freqrawLFPc, meanLFP1PSDc, 'r')
inset.set_xscale('log')
inset.set_yscale('log')
inset.tick_params(axis='x', which='major', bottom=True)
inset.tick_params(axis='y', which='major', left=True)
inset.tick_params(axis='x', which='minor', bottom=True)
inset.tick_params(axis='y', which='minor', left=True)
inset.xaxis.set_minor_locator(MultipleLocator(5))
inset.set_xlim(1,100)
autoscale_y(inset,margin=1e-2,multiplier=100)
ylims = inset.get_ylim()
inset.fill_between(freqrawLFPn, meanLFP1PSDn-stdevLFP1PSDn, meanLFP1PSDn+stdevLFP1PSDn,color='k',alpha=0.4)
inset.fill_between(freqrawLFPc, meanLFP1PSDc-stdevLFP1PSDc, meanLFP1PSDc+stdevLFP1PSDc,color='r',alpha=0.4)
inset.set_ylim(ylims)
ax.set_xlim(0,40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('LFP PSD')
fig.savefig('figs/LFP_PSD.pdf',bbox_inches='tight',transparent=True)
plt.close(fig)

labels = 'PN (70%)', 'MN (15%)', 'BN (10%)', 'VN (5%)'
sizes = [700, 150, 100, 50]
colors = ['dimgrey', 'crimson', 'green', 'orange']
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, colors=colors,
		shadow=False, startangle=90, textprops={'fontsize':20,'weight':'bold'})

wedges = [patch for patch in ax1.patches if isinstance(patch, matplotlib.patches.Wedge)]
for w in wedges:
	w.set_linewidth(0)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

fig1.savefig('figs/CellNumbers.pdf',bbox_inches='tight',transparent=True)
plt.close(fig1)
