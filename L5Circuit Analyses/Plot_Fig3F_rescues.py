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
import seaborn as sns
sns.set_style("white")
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
import LFPy
import pandas as pd

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

N_seeds = 10
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
dt = 0.025
tstop = 7000
startsclice = 2000
endsclice = 6500
t1 = int(startsclice*(1/dt))
t2 = int(endsclice*(1/dt))
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
condit1 = '1_ot2_'+condition+'_0'
condit2 = '1_ot1_'+condition+'_0'
normalpath = normal + '/HL5netLFPy/Circuit_output/'
conditpath = condit + '/HL5netLFPy/Circuit_output/'
conditpath1 = condit1 + '/HL5netLFPy/Circuit_output/'
conditpath2 = condit2 + '/HL5netLFPy/Circuit_output/'

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

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
	
	lmax = np.max(height)
	lmaxind = np.argmax(height)
	
	lx, ly = center[num1], lmax
	rx, ry = center[num2], lmax

	if yerr:
		ly += yerr[lmaxind]
		ry += yerr[lmaxind]

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
conditrates1 = []
conditrates2 = []
normalratesSE = []
conditratesSE = []
conditratesSE1 = []
conditratesSE2 = []
normalpercentsilent = []
conditpercentsilent = []
conditpercentsilent1 = []
conditpercentsilent2 = []

normaleeg = []
conditeeg = []
conditeeg1 = []
conditeeg2 = []

normallfp1 = []
conditlfp1 = []
conditlfp11 = []
conditlfp12 = []

for i in N_seedsList:
	temp_rn = np.loadtxt(normalpath + 'spikerates_Seed' + str(i) + '.txt')
	temp_rc = np.loadtxt(conditpath + 'spikerates_Seed' + str(i) + '.txt')
	temp_rc1 = np.loadtxt(conditpath1 + 'spikerates_Seed' + str(i) + '.txt')
	temp_rc2 = np.loadtxt(conditpath2 + 'spikerates_Seed' + str(i) + '.txt')
	temp_rnSE = np.loadtxt(normalpath + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	temp_rcSE = np.loadtxt(conditpath + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	temp_rcSE1 = np.loadtxt(conditpath1 + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	temp_rcSE2 = np.loadtxt(conditpath2 + 'spikerates_SilentNeuronsExcluded_Seed' + str(i) + '.txt')
	temp_rnS = np.loadtxt(normalpath + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	temp_rcS = np.loadtxt(conditpath + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	temp_rcS1 = np.loadtxt(conditpath1 + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	temp_rcS2 = np.loadtxt(conditpath2 + 'spikerates_PERCENTSILENT_Seed' + str(i) + '.txt')
	temp_en = np.load(normalpath + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ec = np.load(conditpath + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ec01 = np.load(conditpath1 + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ec02 = np.load(conditpath2 + 'DIPOLEMOMENT_Seed' + str(i) + '.npy')
	temp_ln = np.load(normalpath + 'OUTPUT_Seed' + str(i) + '.npy')
	temp_lc = np.load(conditpath + 'OUTPUT_Seed' + str(i) + '.npy')
	temp_lc01 = np.load(conditpath1 + 'OUTPUT_Seed' + str(i) + '.npy')
	temp_lc02 = np.load(conditpath2 + 'OUTPUT_Seed' + str(i) + '.npy')
	
	temp_en2 = temp_en['HL5PN1']
	temp_en2 = np.add(temp_en2,temp_en['HL5MN1'])
	temp_en2 = np.add(temp_en2,temp_en['HL5BN1'])
	temp_en2 = np.add(temp_en2,temp_en['HL5VN1'])
	
	temp_ec2 = temp_ec['HL5PN1']
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5MN1'])
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5BN1'])
	temp_ec2 = np.add(temp_ec2,temp_ec['HL5VN1'])
	
	temp_ec21 = temp_ec01['HL5PN1']
	temp_ec21 = np.add(temp_ec21,temp_ec01['HL5MN1'])
	temp_ec21 = np.add(temp_ec21,temp_ec01['HL5BN1'])
	temp_ec21 = np.add(temp_ec21,temp_ec01['HL5VN1'])
	
	temp_ec22 = temp_ec02['HL5PN1']
	temp_ec22 = np.add(temp_ec22,temp_ec02['HL5MN1'])
	temp_ec22 = np.add(temp_ec22,temp_ec02['HL5BN1'])
	temp_ec22 = np.add(temp_ec22,temp_ec02['HL5VN1'])
	
	potentialn = EEG_args.calc_potential(temp_en2, L5_pos)
	EEGn = potentialn[0][t1:t2]
	potentialc = EEG_args.calc_potential(temp_ec2, L5_pos)
	EEGc = potentialc[0][t1:t2]
	potentialc1 = EEG_args.calc_potential(temp_ec21, L5_pos)
	EEGc1 = potentialc1[0][t1:t2]
	potentialc2 = EEG_args.calc_potential(temp_ec22, L5_pos)
	EEGc2 = potentialc2[0][t1:t2]
	
	freqrawEEGn, psrawEEGn = ss.welch(EEGn, 1/(dt/1000), nperseg=40000)
	freqrawEEGc, psrawEEGc = ss.welch(EEGc, 1/(dt/1000), nperseg=40000)
	freqrawEEGc1, psrawEEGc1 = ss.welch(EEGc1, 1/(dt/1000), nperseg=40000)
	freqrawEEGc2, psrawEEGc2 = ss.welch(EEGc2, 1/(dt/1000), nperseg=40000)
	psrawLFPn = []
	psrawLFPc = []
	psrawLFPc1 = []
	psrawLFPc2 = []
	for l in range(0,len(temp_ln[0])):
		freqrawLFPn, psdn = ss.welch(temp_ln[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=40000)
		freqrawLFPc, psdc = ss.welch(temp_lc[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=40000)
		freqrawLFPc1, psdc1 = ss.welch(temp_lc01[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=40000)
		freqrawLFPc2, psdc2 = ss.welch(temp_lc02[0]['imem'][l][t1:t2], 1/(dt/1000), nperseg=40000)
		psrawLFPn.append(psdn)
		psrawLFPc.append(psdc)
		psrawLFPc1.append(psdc1)
		psrawLFPc2.append(psdc2)
	
	normalrates.append(temp_rn)
	conditrates.append(temp_rc)
	conditrates1.append(temp_rc1)
	conditrates2.append(temp_rc2)
	normalratesSE.append(temp_rnSE)
	conditratesSE.append(temp_rcSE)
	conditratesSE1.append(temp_rcSE1)
	conditratesSE2.append(temp_rcSE2)
	normalpercentsilent.append(temp_rnS)
	conditpercentsilent.append(temp_rcS)
	conditpercentsilent1.append(temp_rcS1)
	conditpercentsilent2.append(temp_rcS2)
	
	normaleeg.append(psrawEEGn)
	conditeeg.append(psrawEEGc)
	conditeeg1.append(psrawEEGc1)
	conditeeg2.append(psrawEEGc2)
	
	normallfp1.append(psrawLFPn[0])
	conditlfp1.append(psrawLFPc[0])
	conditlfp11.append(psrawLFPc1[0])
	conditlfp12.append(psrawLFPc2[0])

meanRatesn = np.mean(normalrates,0)
meanRatesc = np.mean(conditrates,0)
meanRatesc1 = np.mean(conditrates1,0)
meanRatesc2 = np.mean(conditrates2,0)
meanRatesnSE = np.mean(normalratesSE,0)
meanRatescSE = np.mean(conditratesSE,0)
meanRatescSE1 = np.mean(conditratesSE1,0)
meanRatescSE2 = np.mean(conditratesSE2,0)
meanSilentnS = np.mean(normalpercentsilent,0)
meanSilentcS = np.mean(conditpercentsilent,0)
meanSilentcS1 = np.mean(conditpercentsilent1,0)
meanSilentcS2 = np.mean(conditpercentsilent2,0)

stdevRatesn = np.std(normalrates,0)
stdevRatesc = np.std(conditrates,0)
stdevRatesc1 = np.std(conditrates1,0)
stdevRatesc2 = np.std(conditrates2,0)
stdevRatesnSE = np.std(normalratesSE,0)
stdevRatescSE = np.std(conditratesSE,0)
stdevRatescSE1 = np.std(conditratesSE1,0)
stdevRatescSE2 = np.std(conditratesSE2,0)
stdevSilentnS = np.std(normalpercentsilent,0)
stdevSilentcS = np.std(conditpercentsilent,0)
stdevSilentcS1 = np.std(conditpercentsilent1,0)
stdevSilentcS2 = np.std(conditpercentsilent2,0)

meanRates = [meanRatesn[0],meanRatesc[0],meanRatesc1[0],meanRatesc2[0],meanRatesn[1],meanRatesc[1],meanRatesc1[1],meanRatesc2[1],meanRatesn[2],meanRatesc[2],meanRatesc1[2],meanRatesc2[2],meanRatesn[3],meanRatesc[3],meanRatesc1[3],meanRatesc2[3]]
meanRatesSE = [meanRatesnSE[0],meanRatescSE[0],meanRatescSE1[0],meanRatescSE2[0],meanRatesnSE[1],meanRatescSE[1],meanRatescSE1[1],meanRatescSE2[1],meanRatesnSE[2],meanRatescSE[2],meanRatescSE1[2],meanRatescSE2[2],meanRatesnSE[3],meanRatescSE[3],meanRatescSE1[3],meanRatescSE2[3]]
meanSilent = [meanSilentnS[0],meanSilentcS[0],meanSilentcS1[0],meanSilentcS2[0],meanSilentnS[1],meanSilentcS[1],meanSilentcS1[1],meanSilentcS2[1],meanSilentnS[2],meanSilentcS[2],meanSilentcS1[2],meanSilentcS2[2],meanSilentnS[3],meanSilentcS[3],meanSilentcS1[3],meanSilentcS2[3]]

stdevRate = [stdevRatesn[0],stdevRatesc[0],stdevRatesc1[0],stdevRatesc2[0],stdevRatesn[1],stdevRatesc[1],stdevRatesc1[1],stdevRatesc2[1],stdevRatesn[2],stdevRatesc[2],stdevRatesc1[2],stdevRatesc2[2],stdevRatesn[3],stdevRatesc[3],stdevRatesc1[3],stdevRatesc2[3]]
stdevRateSE = [stdevRatesnSE[0],stdevRatescSE[0],stdevRatescSE1[0],stdevRatescSE2[0],stdevRatesnSE[1],stdevRatescSE[1],stdevRatescSE1[1],stdevRatescSE2[1],stdevRatesnSE[2],stdevRatescSE[2],stdevRatescSE1[2],stdevRatescSE2[2],stdevRatesnSE[3],stdevRatescSE[3],stdevRatescSE1[3],stdevRatescSE2[3]]
stdevSilent = [stdevSilentnS[0],stdevSilentcS[0],stdevSilentcS1[0],stdevSilentcS2[0],stdevSilentnS[1],stdevSilentcS[1],stdevSilentcS1[1],stdevSilentcS2[1],stdevSilentnS[2],stdevSilentcS[2],stdevSilentcS1[2],stdevSilentcS2[2],stdevSilentnS[3],stdevSilentcS[3],stdevSilentcS1[3],stdevSilentcS2[3]]

nr = np.array(normalrates).transpose()
nc = np.array(conditrates).transpose()
nc1 = np.array(conditrates1).transpose()
nc2 = np.array(conditrates2).transpose()
Rates = [nr[0],nc[0],nc1[0],nc2[0],nr[1],nc[1],nc1[1],nc2[1],nr[2],nc[2],nc1[2],nc2[2],nr[3],nc[3],nc1[3],nc2[3]]

nrSE = np.array(normalratesSE).transpose()
ncSE = np.array(conditratesSE).transpose()
ncSE1 = np.array(conditratesSE1).transpose()
ncSE2 = np.array(conditratesSE2).transpose()
RatesSE = [nrSE[0],ncSE[0],ncSE1[0],ncSE2[0],nrSE[1],ncSE[1],ncSE1[1],ncSE2[1],nrSE[2],ncSE[2],ncSE1[2],ncSE2[2],nrSE[3],ncSE[3],ncSE1[3],ncSE2[3]]

nrS = np.array(normalpercentsilent).transpose()
ncS = np.array(conditpercentsilent).transpose()
ncS1 = np.array(conditpercentsilent1).transpose()
ncS2 = np.array(conditpercentsilent2).transpose()
Silent = [nrS[0],ncS[0],ncS1[0],ncS2[0],nrS[1],ncS[1],ncS1[1],ncS2[1],nrS[2],ncS[2],ncS1[2],ncS2[2],nrS[3],ncS[3],ncS1[3],ncS2[3]]

tstat_PN, pval_PN = st.ttest_rel(nr[0],nc[0])
tstat_MN, pval_MN = st.ttest_rel(nr[1],nc[1])
tstat_BN, pval_BN = st.ttest_rel(nr[2],nc[2])
tstat_VN, pval_VN = st.ttest_rel(nr[3],nc[3])

tstat_PNSE, pval_PNSE = st.ttest_rel(nrSE[0],ncSE[0])
tstat_MNSE, pval_MNSE = st.ttest_rel(nrSE[1],ncSE[1])
tstat_BNSE, pval_BNSE = st.ttest_rel(nrSE[2],ncSE[2])
tstat_VNSE, pval_VNSE = st.ttest_rel(nrSE[3],ncSE[3])

tstat_PNSE1, pval_PNSE1 = st.ttest_rel(nrSE[0],ncSE1[0])
tstat_MNSE1, pval_MNSE1 = st.ttest_rel(nrSE[1],ncSE1[1])
tstat_BNSE1, pval_BNSE1 = st.ttest_rel(nrSE[2],ncSE1[2])
tstat_VNSE1, pval_VNSE1 = st.ttest_rel(nrSE[3],ncSE1[3])

tstat_PNSE2, pval_PNSE2 = st.ttest_rel(nrSE[0],ncSE2[0])
tstat_MNSE2, pval_MNSE2 = st.ttest_rel(nrSE[1],ncSE2[1])
tstat_BNSE2, pval_BNSE2 = st.ttest_rel(nrSE[2],ncSE2[2])
tstat_VNSE2, pval_VNSE2 = st.ttest_rel(nrSE[3],ncSE2[3])

tstat_PNS, pval_PNS = st.ttest_rel(nrS[0],ncS[0])
tstat_MNS, pval_MNS = st.ttest_rel(nrS[1],ncS[1])
tstat_BNS, pval_BNS = st.ttest_rel(nrS[2],ncS[2])
tstat_VNS, pval_VNS = st.ttest_rel(nrS[3],ncS[3])

meanRates_PNonly = [meanRatesn[0],meanRatesc[0],meanRatesc1[0],meanRatesc2[0]]
stdevRate_PNonly = [stdevRatesn[0],stdevRatesc[0],stdevRatesc1[0],stdevRatesc2[0]]
Rates_PNonly = [nr[0],nc[0],nc1[0],nc2[0]]

meanRates_PNonlySE = [meanRatesnSE[0],meanRatescSE[0],meanRatescSE1[0],meanRatescSE2[0]]
stdevRate_PNonlySE = [stdevRatesnSE[0],stdevRatescSE[0],stdevRatescSE1[0],stdevRatescSE2[0]]
Rates_PNonlySE = [nrSE[0],ncSE[0],ncSE1[0],ncSE2[0]]

meanSilent_PNonly = [meanSilentnS[0],meanSilentcS[0],meanSilentcS1[0],meanSilentcS2[0]]
stdevSilent_PNonly = [stdevSilentnS[0],stdevSilentcS[0],stdevSilentcS1[0],stdevSilentcS2[0]]
Silent_PNonlyS = [nrS[0],ncS[0],ncS1[0],ncS2[0]]

df = pd.DataFrame(columns=["Mean Young",
			"SD Young",
			"Mean Old",
			"SD Old",
			"Mean Old + Ydend",
			"SD Old + Ydend",
			"Mean Old + YH",
			"SD Old + YH",
			"t-stat (Y vs O)",
			"p-value (Y vs O)",
			"Cohen's d (Y vs O)",
			"t-stat (Y vs O+Ydend)",
			"p-value (Y vs O+Ydend)",
			"Cohen's d (Y vs O+Ydend)",
			"t-stat (Y vs O+YH)",
			"p-value (Y vs O+YH)",
			"Cohen's d (Y vs O+YH)"])

# Calculate Cohen's d for PN spike rate with silent excluded for each condition
cd = cohen_d(nrSE[0],ncSE[0])
cd1 = cohen_d(nrSE[0],ncSE1[0])
cd2 = cohen_d(nrSE[0],ncSE2[0])

df = df.append({"Mean Young" : meanRatesnSE[0],
			"SD Young" : stdevRatesnSE[0],
			"Mean Old" : meanRatescSE[0],
			"SD Old" : stdevRatescSE[0],
			"Mean Old + Ydend" : meanRatescSE1[0],
			"SD Old + Ydend" : stdevRatescSE1[0],
			"Mean Old + YH" : meanRatescSE2[0],
			"SD Old + YH" : stdevRatescSE2[0],
			"t-stat (Y vs O)" : tstat_PNSE,
			"p-value (Y vs O)" : pval_PNSE,
			"Cohen's d (Y vs O)" : cd,
			"t-stat (Y vs O+Ydend)" : tstat_PNSE1,
			"p-value (Y vs O+Ydend)" : pval_PNSE1,
			"Cohen's d (Y vs O+Ydend)" : cd1,
			"t-stat (Y vs O+YH)" : tstat_PNSE2,
			"p-value (Y vs O+YH)" : pval_PNSE2,
			"Cohen's d (Y vs O+YH)" : cd2},
			ignore_index = True)

df.to_csv('figs_alltests/stats_baseline_AllConditions.csv')

meanstdevstr1n = '\n' + str(np.around(meanRatesn[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesn[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesn[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesn[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanRatesc[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanRatesc[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanRatesc[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanRatesc[3], decimals=2))
meanstdevstr1c1 = '\n' + str(np.around(meanRatesc1[0], decimals=2))
meanstdevstr2c1 = '\n' + str(np.around(meanRatesc1[1], decimals=2))
meanstdevstr3c1 = '\n' + str(np.around(meanRatesc1[2], decimals=2))
meanstdevstr4c1 = '\n' + str(np.around(meanRatesc1[3], decimals=2))
meanstdevstr1c2 = '\n' + str(np.around(meanRatesc2[0], decimals=2))
meanstdevstr2c2 = '\n' + str(np.around(meanRatesc2[1], decimals=2))
meanstdevstr3c2 = '\n' + str(np.around(meanRatesc2[2], decimals=2))
meanstdevstr4c2 = '\n' + str(np.around(meanRatesc2[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
namesc1 = ['PN'+meanstdevstr1c1,'MN'+meanstdevstr2c1,'BM'+meanstdevstr3c1,'VN'+meanstdevstr4c1]
namesc2 = ['PN'+meanstdevstr1c2,'MN'+meanstdevstr2c2,'BM'+meanstdevstr3c2,'VN'+meanstdevstr4c2]
names = [namesn[0],namesc[0],namesc1[0],namesc2[0],namesn[1],namesc[1],namesc1[1],namesc2[1],namesn[2],namesc[2],namesc1[2],namesc2[2],namesn[3],namesc[3],namesc1[3],namesc2[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15]
colors = ['dimgray', 'dimgray', 'dimgray', 'dimgray', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green', 'orange', 'orange', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRates,
	   yerr=stdevRate,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.4,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

# if pval_PN < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', x, meanRates, yerr=stdevRate)
# elif pval_PN < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', x, meanRates, yerr=stdevRate)
# elif pval_PN < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', x, meanRates, yerr=stdevRate)
#
# if pval_MN < 0.001:
# 	barplot_annotate_brackets(2, 3, '***', x, meanRates, yerr=stdevRate)
# elif pval_MN < 0.01:
# 	barplot_annotate_brackets(2, 3, '**', x, meanRates, yerr=stdevRate)
# elif pval_MN < 0.05:
# 	barplot_annotate_brackets(2, 3, '*', x, meanRates, yerr=stdevRate)
#
# if pval_BN < 0.001:
# 	barplot_annotate_brackets(4, 5, '***', x, meanRates, yerr=stdevRate)
# elif pval_BN < 0.01:
# 	barplot_annotate_brackets(4, 5, '**', x, meanRates, yerr=stdevRate)
# elif pval_BN < 0.05:
# 	barplot_annotate_brackets(4, 5, '*', x, meanRates, yerr=stdevRate)
#
# if pval_VN < 0.001:
# 	barplot_annotate_brackets(6, 7, '***', x, meanRates, yerr=stdevRate)
# elif pval_VN < 0.01:
# 	barplot_annotate_brackets(6, 7, '**', x, meanRates, yerr=stdevRate)
# elif pval_VN < 0.05:
# 	barplot_annotate_brackets(6, 7, '*', x, meanRates, yerr=stdevRate)

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/Rates.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesnSE[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanRatesnSE[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanRatesnSE[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanRatesnSE[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanRatescSE[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanRatescSE[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanRatescSE[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanRatescSE[3], decimals=2))
meanstdevstr1c1 = '\n' + str(np.around(meanRatescSE1[0], decimals=2))
meanstdevstr2c1 = '\n' + str(np.around(meanRatescSE1[1], decimals=2))
meanstdevstr3c1 = '\n' + str(np.around(meanRatescSE1[2], decimals=2))
meanstdevstr4c1 = '\n' + str(np.around(meanRatescSE1[3], decimals=2))
meanstdevstr1c2 = '\n' + str(np.around(meanRatescSE2[0], decimals=2))
meanstdevstr2c2 = '\n' + str(np.around(meanRatescSE2[1], decimals=2))
meanstdevstr3c2 = '\n' + str(np.around(meanRatescSE2[2], decimals=2))
meanstdevstr4c2 = '\n' + str(np.around(meanRatescSE2[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
namesc1 = ['PN'+meanstdevstr1c1,'MN'+meanstdevstr2c1,'BM'+meanstdevstr3c1,'VN'+meanstdevstr4c1]
namesc2 = ['PN'+meanstdevstr1c2,'MN'+meanstdevstr2c2,'BM'+meanstdevstr3c2,'VN'+meanstdevstr4c2]
names = [namesn[0],namesc[0],namesc1[0],namesc2[0],namesn[1],namesc[1],namesc1[1],namesc2[1],namesn[2],namesc[2],namesc1[2],namesc2[2],namesn[3],namesc[3],namesc1[3],namesc2[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15]
colors = ['dimgray', 'dimgray', 'dimgray', 'dimgray', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green', 'orange', 'orange', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanRatesSE,
	   yerr=stdevRateSE,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.4,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

# if pval_PNSE < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_PNSE < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_PNSE < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', x, meanRatesSE, yerr=stdevRateSE)
#
# if pval_MNSE < 0.001:
# 	barplot_annotate_brackets(2, 3, '***', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_MNSE < 0.01:
# 	barplot_annotate_brackets(2, 3, '**', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_MNSE < 0.05:
# 	barplot_annotate_brackets(2, 3, '*', x, meanRatesSE, yerr=stdevRateSE)
#
# if pval_BNSE < 0.001:
# 	barplot_annotate_brackets(4, 5, '***', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_BNSE < 0.01:
# 	barplot_annotate_brackets(4, 5, '**', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_BNSE < 0.05:
# 	barplot_annotate_brackets(4, 5, '*', x, meanRatesSE, yerr=stdevRateSE)
#
# if pval_VNSE < 0.001:
# 	barplot_annotate_brackets(6, 7, '***', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_VNSE < 0.01:
# 	barplot_annotate_brackets(6, 7, '**', x, meanRatesSE, yerr=stdevRateSE)
# elif pval_VNSE < 0.05:
# 	barplot_annotate_brackets(6, 7, '*', x, meanRatesSE, yerr=stdevRateSE)

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/RatesSE.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanSilentnS[0], decimals=2))
meanstdevstr2n = '\n' + str(np.around(meanSilentnS[1], decimals=2))
meanstdevstr3n = '\n' + str(np.around(meanSilentnS[2], decimals=2))
meanstdevstr4n = '\n' + str(np.around(meanSilentnS[3], decimals=2))
meanstdevstr1c = '\n' + str(np.around(meanSilentcS[0], decimals=2))
meanstdevstr2c = '\n' + str(np.around(meanSilentcS[1], decimals=2))
meanstdevstr3c = '\n' + str(np.around(meanSilentcS[2], decimals=2))
meanstdevstr4c = '\n' + str(np.around(meanSilentcS[3], decimals=2))
meanstdevstr1c1 = '\n' + str(np.around(meanSilentcS1[0], decimals=2))
meanstdevstr2c1 = '\n' + str(np.around(meanSilentcS1[1], decimals=2))
meanstdevstr3c1 = '\n' + str(np.around(meanSilentcS1[2], decimals=2))
meanstdevstr4c1 = '\n' + str(np.around(meanSilentcS1[3], decimals=2))
meanstdevstr1c2 = '\n' + str(np.around(meanSilentcS2[0], decimals=2))
meanstdevstr2c2 = '\n' + str(np.around(meanSilentcS2[1], decimals=2))
meanstdevstr3c2 = '\n' + str(np.around(meanSilentcS2[2], decimals=2))
meanstdevstr4c2 = '\n' + str(np.around(meanSilentcS2[3], decimals=2))
namesn = ['PN'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['PN'+meanstdevstr1c,'MN'+meanstdevstr2c,'BM'+meanstdevstr3c,'VN'+meanstdevstr4c]
namesc1 = ['PN'+meanstdevstr1c1,'MN'+meanstdevstr2c1,'BM'+meanstdevstr3c1,'VN'+meanstdevstr4c1]
namesc2 = ['PN'+meanstdevstr1c2,'MN'+meanstdevstr2c2,'BM'+meanstdevstr3c2,'VN'+meanstdevstr4c2]
names = [namesn[0],namesc[0],namesc1[0],namesc2[0],namesn[1],namesc[1],namesc1[1],namesc2[1],namesn[2],namesc[2],namesc1[2],namesc2[2],namesn[3],namesc[3],namesc1[3],namesc2[3]]

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15]
colors = ['dimgray', 'dimgray', 'dimgray', 'dimgray', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green', 'orange', 'orange', 'orange', 'orange']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
ax.bar(x,height=meanSilent,
	   yerr=stdevSilent,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.4,    # bar width
	   tick_label=names,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

# if pval_PNS < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', x, meanSilent, yerr=stdevSilent)
# elif pval_PNS < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', x, meanSilent, yerr=stdevSilent)
# elif pval_PNS < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', x, meanSilent, yerr=stdevSilent)
#
# if pval_MNS < 0.001:
# 	barplot_annotate_brackets(2, 3, '***', x, meanSilent, yerr=stdevSilent)
# elif pval_MNS < 0.01:
# 	barplot_annotate_brackets(2, 3, '**', x, meanSilent, yerr=stdevSilent)
# elif pval_MNS < 0.05:
# 	barplot_annotate_brackets(2, 3, '*', x, meanSilent, yerr=stdevSilent)
#
# if pval_BNS < 0.001:
# 	barplot_annotate_brackets(4, 5, '***', x, meanSilent, yerr=stdevSilent)
# elif pval_BNS < 0.01:
# 	barplot_annotate_brackets(4, 5, '**', x, meanSilent, yerr=stdevSilent)
# elif pval_BNS < 0.05:
# 	barplot_annotate_brackets(4, 5, '*', x, meanSilent, yerr=stdevSilent)
#
# if pval_VNS < 0.001:
# 	barplot_annotate_brackets(6, 7, '***', x, meanSilent, yerr=stdevSilent)
# elif pval_VNS < 0.01:
# 	barplot_annotate_brackets(6, 7, '**', x, meanSilent, yerr=stdevSilent)
# elif pval_VNS < 0.05:
# 	barplot_annotate_brackets(6, 7, '*', x, meanSilent, yerr=stdevSilent)

ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/SilentAll.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesn[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[0], decimals=2)) + ' Hz'
meanstdevstr2n = '\n' + str(np.around(meanRatesn[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[1], decimals=2)) + ' Hz'
meanstdevstr3n = '\n' + str(np.around(meanRatesn[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[2], decimals=2)) + ' Hz'
meanstdevstr4n = '\n' + str(np.around(meanRatesn[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesn[3], decimals=2)) + ' Hz'
meanstdevstr1c = '\n' + str(np.around(meanRatesc[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[0], decimals=2)) + ' Hz'
meanstdevstr2c = '\n' + str(np.around(meanRatesc[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[1], decimals=2)) + ' Hz'
meanstdevstr3c = '\n' + str(np.around(meanRatesc[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[2], decimals=2)) + ' Hz'
meanstdevstr4c = '\n' + str(np.around(meanRatesc[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc[3], decimals=2)) + ' Hz'
meanstdevstr1c1 = '\n' + str(np.around(meanRatesc1[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc1[0], decimals=2)) + ' Hz'
meanstdevstr2c1 = '\n' + str(np.around(meanRatesc1[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc1[1], decimals=2)) + ' Hz'
meanstdevstr3c1 = '\n' + str(np.around(meanRatesc1[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc1[2], decimals=2)) + ' Hz'
meanstdevstr4c1 = '\n' + str(np.around(meanRatesc1[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc1[3], decimals=2)) + ' Hz'
meanstdevstr1c2 = '\n' + str(np.around(meanRatesc2[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc2[0], decimals=2)) + ' Hz'
meanstdevstr2c2 = '\n' + str(np.around(meanRatesc2[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc2[1], decimals=2)) + ' Hz'
meanstdevstr3c2 = '\n' + str(np.around(meanRatesc2[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc2[2], decimals=2)) + ' Hz'
meanstdevstr4c2 = '\n' + str(np.around(meanRatesc2[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesc2[3], decimals=2)) + ' Hz'

namesn = ['Young'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['Old'+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
namesc1 = ['Old T1'+meanstdevstr1c1,manipulation+meanstdevstr2c1,manipulation+meanstdevstr3c1,manipulation+meanstdevstr4c1]
namesc2 = ['Old T2'+meanstdevstr1c2,manipulation+meanstdevstr2c2,manipulation+meanstdevstr3c2,manipulation+meanstdevstr4c2]
names_PNonly = [namesn[0],namesc[0],namesc1[0],namesc2[0]]
x = [0, 1, 2, 3]
colors = ['darkgray', 'lightcoral', 'lightcoral', 'lightcoral']
hatches = ['','','/','//']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
bars = ax.bar(x,height=meanRates_PNonly,
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

for bar, pattern in zip(bars, hatches):
	bar.set_hatch(pattern)

# if pval_PN < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', x, meanRates_PNonly, yerr=stdevRate_PNonly)
# elif pval_PN < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', x, meanRates_PNonly, yerr=stdevRate_PNonly)
# elif pval_PN < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', x, meanRates_PNonly, yerr=stdevRate_PNonly)

ax.set_xlim(-0.65,3.65)
ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/Rates_PNonly.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanRatesnSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[0], decimals=2)) + ' Hz'
meanstdevstr2n = '\n' + str(np.around(meanRatesnSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[1], decimals=2)) + ' Hz'
meanstdevstr3n = '\n' + str(np.around(meanRatesnSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[2], decimals=2)) + ' Hz'
meanstdevstr4n = '\n' + str(np.around(meanRatesnSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatesnSE[3], decimals=2)) + ' Hz'
meanstdevstr1c = '\n' + str(np.around(meanRatescSE[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[0], decimals=2)) + ' Hz'
meanstdevstr2c = '\n' + str(np.around(meanRatescSE[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[1], decimals=2)) + ' Hz'
meanstdevstr3c = '\n' + str(np.around(meanRatescSE[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[2], decimals=2)) + ' Hz'
meanstdevstr4c = '\n' + str(np.around(meanRatescSE[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE[3], decimals=2)) + ' Hz'
meanstdevstr1c1 = '\n' + str(np.around(meanRatescSE1[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE1[0], decimals=2)) + ' Hz'
meanstdevstr2c1 = '\n' + str(np.around(meanRatescSE1[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE1[1], decimals=2)) + ' Hz'
meanstdevstr3c1 = '\n' + str(np.around(meanRatescSE1[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE1[2], decimals=2)) + ' Hz'
meanstdevstr4c1 = '\n' + str(np.around(meanRatescSE1[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE1[3], decimals=2)) + ' Hz'
meanstdevstr1c2 = '\n' + str(np.around(meanRatescSE2[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE2[0], decimals=2)) + ' Hz'
meanstdevstr2c2 = '\n' + str(np.around(meanRatescSE2[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE2[1], decimals=2)) + ' Hz'
meanstdevstr3c2 = '\n' + str(np.around(meanRatescSE2[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE2[2], decimals=2)) + ' Hz'
meanstdevstr4c2 = '\n' + str(np.around(meanRatescSE2[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevRatescSE2[3], decimals=2)) + ' Hz'
namesn = ['Young'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['Old'+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
namesc1 = ['Old T1'+meanstdevstr1c1,manipulation+meanstdevstr2c1,manipulation+meanstdevstr3c1,manipulation+meanstdevstr4c1]
namesc2 = ['Old T2'+meanstdevstr1c2,manipulation+meanstdevstr2c2,manipulation+meanstdevstr3c2,manipulation+meanstdevstr4c2]
names_PNonly = [namesn[0],namesc[0],namesc1[0],namesc2[0]]
x = [0, 1, 2, 3]
colors = ['darkgray', 'lightcoral', 'lightcoral', 'lightcoral']
hatches = ['','','/','//']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
bars = ax.bar(x,height=meanRates_PNonlySE,
	   yerr=stdevRate_PNonlySE,    # error bars
	   capsize=12, # error bar cap width in points
	   width=0.8,    # bar width
	   tick_label=names_PNonly,
	   color=[clr for clr in colors],  # face color transparent
	   edgecolor='k',
	   ecolor='black',
	   linewidth=1,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

for bar, pattern in zip(bars, hatches):
	bar.set_hatch(pattern)

if pval_PNSE < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.05)
elif pval_PNSE < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.05)
elif pval_PNSE < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.05)

if pval_PNSE1 < 0.001:
	barplot_annotate_brackets(0, 2, '***', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.1)
elif pval_PNSE1 < 0.01:
	barplot_annotate_brackets(0, 2, '**', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.1)
elif pval_PNSE1 < 0.05:
	barplot_annotate_brackets(0, 2, '*', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.1)

if pval_PNSE2 < 0.001:
	barplot_annotate_brackets(0, 3, '***', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.15)
elif pval_PNSE2 < 0.01:
	barplot_annotate_brackets(0, 3, '**', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.15)
elif pval_PNSE2 < 0.05:
	barplot_annotate_brackets(0, 3, '*', x, meanRates_PNonlySE, yerr=stdevRate_PNonlySE,dh=0.15)

ax.set_xlim(-0.65,3.65)
ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/Rates_PNonlySE.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanstdevstr1n = '\n' + str(np.around(meanSilentnS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[0], decimals=2)) + ' %'
meanstdevstr2n = '\n' + str(np.around(meanSilentnS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[1], decimals=2)) + ' %'
meanstdevstr3n = '\n' + str(np.around(meanSilentnS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[2], decimals=2)) + ' %'
meanstdevstr4n = '\n' + str(np.around(meanSilentnS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentnS[3], decimals=2)) + ' %'
meanstdevstr1c = '\n' + str(np.around(meanSilentcS[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[0], decimals=2)) + ' %'
meanstdevstr2c = '\n' + str(np.around(meanSilentcS[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[1], decimals=2)) + ' %'
meanstdevstr3c = '\n' + str(np.around(meanSilentcS[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[2], decimals=2)) + ' %'
meanstdevstr4c = '\n' + str(np.around(meanSilentcS[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS[3], decimals=2)) + ' %'
meanstdevstr1c1 = '\n' + str(np.around(meanSilentcS1[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS1[0], decimals=2)) + ' %'
meanstdevstr2c1 = '\n' + str(np.around(meanSilentcS1[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS1[1], decimals=2)) + ' %'
meanstdevstr3c1 = '\n' + str(np.around(meanSilentcS1[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS1[2], decimals=2)) + ' %'
meanstdevstr4c1 = '\n' + str(np.around(meanSilentcS1[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS1[3], decimals=2)) + ' %'
meanstdevstr1c2 = '\n' + str(np.around(meanSilentcS2[0], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS2[0], decimals=2)) + ' %'
meanstdevstr2c2 = '\n' + str(np.around(meanSilentcS2[1], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS2[1], decimals=2)) + ' %'
meanstdevstr3c2 = '\n' + str(np.around(meanSilentcS2[2], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS2[2], decimals=2)) + ' %'
meanstdevstr4c2 = '\n' + str(np.around(meanSilentcS2[3], decimals=2)) + r' $\pm$ '+ str(np.around(stdevSilentcS2[3], decimals=2)) + ' %'
namesn = ['Young'+meanstdevstr1n,'MN'+meanstdevstr2n,'BN'+meanstdevstr3n,'VN'+meanstdevstr4n]
namesc = ['Old'+meanstdevstr1c,manipulation+meanstdevstr2c,manipulation+meanstdevstr3c,manipulation+meanstdevstr4c]
namesc1 = ['Old T1'+meanstdevstr1c1,manipulation+meanstdevstr2c1,manipulation+meanstdevstr3c1,manipulation+meanstdevstr4c1]
namesc2 = ['Old T2'+meanstdevstr1c2,manipulation+meanstdevstr2c2,manipulation+meanstdevstr3c2,manipulation+meanstdevstr4c2]
names_PNonly = [namesn[0],namesc[0],namesc1[0],namesc2[0]]
x = [0, 1, 2, 3]
colors = ['dimgray', 'red', 'red', 'red']
hatches = ['','','/','//']
fig, ax = plt.subplots(figsize=(8, 8))
# sns.swarmplot(data=np.array(Rates_PNonly).transpose(),palette=colors, alpha=0.6, linewidth=1, edgecolor='k')
bars = ax.bar(x,height=meanSilent_PNonly,
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

for bar, pattern in zip(bars, hatches):
	bar.set_hatch(pattern)

# if pval_PNS < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)
# elif pval_PNS < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)
# elif pval_PNS < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', x, meanSilent_PNonly, yerr=stdevSilent_PNonly)

ax.set_xlim(-0.65,3.65)
ax.set_ylabel('Percent Silent (%)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/PercentSilent_PNonly.pdf',bbox_inches='tight', dpi=300, transparent=True)
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
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/RatesNormal.pdf',bbox_inches='tight', dpi=300, transparent=True)
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
	   linewidth=4,
	   error_kw={'elinewidth':3,'markeredgewidth':3}
	  )

ax.set_ylabel('Spike Rate (Hz)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig('figs_alltests/RatesNormalSE.pdf',bbox_inches='tight', dpi=300, transparent=True)
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
fig.savefig('figs_alltests/SilentNormal.pdf',bbox_inches='tight', dpi=300, transparent=True)
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
fig.savefig('figs_alltests/SilentNormal_PNOnly.pdf',bbox_inches='tight', dpi=300, transparent=True)
plt.close()

meanEEGPSDn = np.mean(normaleeg,0)
meanEEGPSDc = np.mean(conditeeg,0)
meanEEGPSDc1 = np.mean(conditeeg1,0)
meanEEGPSDc2 = np.mean(conditeeg2,0)
stdevEEGPSDn = np.std(normaleeg,0)
stdevEEGPSDc = np.std(conditeeg,0)
stdevEEGPSDc1 = np.std(conditeeg1,0)
stdevEEGPSDc2 = np.std(conditeeg2,0)

meanLFP1PSDn = np.mean(normallfp1,0)
meanLFP1PSDc = np.mean(conditlfp1,0)
meanLFP1PSDc1 = np.mean(conditlfp11,0)
meanLFP1PSDc2 = np.mean(conditlfp12,0)
stdevLFP1PSDn = np.std(normallfp1,0)
stdevLFP1PSDc = np.std(conditlfp1,0)
stdevLFP1PSDc1 = np.std(conditlfp11,0)
stdevLFP1PSDc2 = np.std(conditlfp12,0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(freqrawEEGn, meanEEGPSDn, 'k')
ax.plot(freqrawEEGc, meanEEGPSDc, 'r')
ax.plot(freqrawEEGc1, meanEEGPSDc1, 'r')
ax.plot(freqrawEEGc2, meanEEGPSDc2, 'r')
ax.fill_between(freqrawEEGn, meanEEGPSDn-stdevEEGPSDn, meanEEGPSDn+stdevEEGPSDn,color='k',alpha=0.4)
ax.fill_between(freqrawEEGc, meanEEGPSDc-stdevEEGPSDc, meanEEGPSDc+stdevEEGPSDc,color='r',alpha=0.4)
ax.fill_between(freqrawEEGc1, meanEEGPSDc1-stdevEEGPSDc1, meanEEGPSDc1+stdevEEGPSDc1, facecolor='none',alpha=0.4,hatch="+",edgecolor="k",linewidth=1)
ax.fill_between(freqrawEEGc2, meanEEGPSDc2-stdevEEGPSDc2, meanEEGPSDc2+stdevEEGPSDc2, facecolor='none',alpha=0.4,hatch="XXX",edgecolor="k",linewidth=2)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='y', which='major', left=True)
inset = ax.inset_axes([.6,.6,.35,.35])
inset.plot(freqrawEEGn, meanEEGPSDn, 'k')
inset.plot(freqrawEEGc, meanEEGPSDc, 'r')
inset.plot(freqrawEEGc1, meanEEGPSDc1, 'r')
inset.plot(freqrawEEGc2, meanEEGPSDc2, 'r')
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
inset.fill_between(freqrawEEGc1, meanEEGPSDc1-stdevEEGPSDc1, meanEEGPSDc1+stdevEEGPSDc1,facecolor='none',alpha=0.4,hatch="+",edgecolor="k",linewidth=1)
inset.fill_between(freqrawEEGc2, meanEEGPSDc2-stdevEEGPSDc2, meanEEGPSDc2+stdevEEGPSDc2,facecolor='none',alpha=0.4,hatch="XXX",edgecolor="k",linewidth=2)
inset.set_ylim(ylims)
ax.set_xlim(0,40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('EEG PSD')
fig.savefig('figs_alltests/EEG_PSD.pdf',bbox_inches='tight',dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(freqrawLFPn,meanLFP1PSDn,'k')
ax.plot(freqrawLFPc,meanLFP1PSDc,'r')
ax.plot(freqrawLFPc1,meanLFP1PSDc1,'r')
ax.plot(freqrawLFPc2,meanLFP1PSDc2,'r')
ax.fill_between(freqrawLFPn, meanLFP1PSDn-stdevLFP1PSDn, meanLFP1PSDn+stdevLFP1PSDn,color='k',alpha=0.4)
ax.fill_between(freqrawLFPc, meanLFP1PSDc-stdevLFP1PSDc, meanLFP1PSDc+stdevLFP1PSDc,color='r',alpha=0.4)
ax.fill_between(freqrawLFPc1, meanLFP1PSDc1-stdevLFP1PSDc1, meanLFP1PSDc1+stdevLFP1PSDc1,facecolor='none',alpha=0.4,hatch="+",edgecolor="k",linewidth=2)
ax.fill_between(freqrawLFPc2, meanLFP1PSDc2-stdevLFP1PSDc2, meanLFP1PSDc2+stdevLFP1PSDc2,facecolor='none',alpha=0.4,hatch="XXX",edgecolor="k",linewidth=2)
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
inset.fill_between(freqrawLFPc1, meanLFP1PSDc1-stdevLFP1PSDc1, meanLFP1PSDc1+stdevLFP1PSDc1,facecolor='none',alpha=0.4,hatch="+",edgecolor="k",linewidth=2)
inset.fill_between(freqrawLFPc2, meanLFP1PSDc2-stdevLFP1PSDc2, meanLFP1PSDc2+stdevLFP1PSDc2,facecolor='none',alpha=0.4,hatch="XXX",edgecolor="k",linewidth=2)
inset.set_ylim(ylims)
ax.set_xlim(0,40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('LFP PSD')
fig.savefig('figs_alltests/LFP_PSD.pdf',bbox_inches='tight',transparent=True)
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

fig1.savefig('figs_alltests/CellNumbers.pdf',bbox_inches='tight',transparent=True)
plt.close(fig1)
