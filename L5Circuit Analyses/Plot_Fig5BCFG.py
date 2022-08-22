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
N_HL5PNso = int(N_HL5PN*0.5)

dt = 0.025
tstop = 7000

transient = 2000
stimtime = 6500 # Absolute time during stimulation that is set for stimtime
stimbegin = 3 # Time (ms) after stimtime where stimulation begins (i.e. the system delay)
stimend = 103 # Stimbegin + duration of stimulus
startslice = stimtime+stimbegin
endslice = stimtime+stimend
totalstim = endslice-startslice
tvec = np.arange(startslice,endslice+1,dt,dtype=np.float64)

synnums = [85,95]
N_seeds = 80
rndinds = np.linspace(1,N_seeds,N_seeds, dtype=int)

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

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

def gaussian(x, mu, sig):
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

Ratesy_allseeds1 = [[] for _ in range(0,len(rndinds))]
Rateso_allseeds1 = [[] for _ in range(0,len(rndinds))]
Ratesy_allseeds2 = [[] for _ in range(0,len(rndinds))]
Rateso_allseeds2 = [[] for _ in range(0,len(rndinds))]
SNRy_allseeds1 = [[] for _ in range(0,len(rndinds))]
SNRo_allseeds1 = [[] for _ in range(0,len(rndinds))]
SNRy_allseeds2 = [[] for _ in range(0,len(rndinds))]
SNRo_allseeds2 = [[] for _ in range(0,len(rndinds))]
BaseRatey_allseeds1 = [[] for _ in range(0,len(rndinds))]
BaseRateo_allseeds1 = [[] for _ in range(0,len(rndinds))]
BaseRatey_allseeds2 = [[] for _ in range(0,len(rndinds))]
BaseRateo_allseeds2 = [[] for _ in range(0,len(rndinds))]
for control in controlm:
	if control == 'y': colc = 'k'
	if control == 'o': colc = 'r'
	path1 = control + '_' + str(synnums[0])+ '/HL5netLFPy/Circuit_output/'
	path2 = control + '_' + str(synnums[1])+ '/HL5netLFPy/Circuit_output/'
	
	for idx in range(0,len(rndinds)):
		print('Seed #'+str(idx))
		temp_sn1 = np.load(path1 + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
		temp_sn2 = np.load(path2 + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
		
		# Only analyze PNs here
		SPIKES1 = [x for _,x in sorted(zip(temp_sn1.item()['gids'][0],temp_sn1.item()['times'][0]))]
		SPIKES2 = [x for _,x in sorted(zip(temp_sn2.item()['gids'][0],temp_sn2.item()['times'][0]))]
		for j in range(N_HL5PN):
			tv_base = np.around(np.array(SPIKES1[j][(SPIKES1[j]>transient) & (SPIKES1[j]<stimtime)]),3)
			rate_base = (len(tv_base)*1000)/(stimtime-transient) # convert to 1/seconds
			tv = np.around(np.array(SPIKES1[j][(SPIKES1[j]>startslice) & (SPIKES1[j]<endslice)]),3)
			rate = (len(tv)*1000)/(endslice-startslice) # convert to 1/seconds
			SNR = rate/rate_base if (rate_base > 0) else 0
			if control == 'y': Ratesy_allseeds1[idx].append(rate)
			if control == 'o': Rateso_allseeds1[idx].append(rate)
			if control == 'y': BaseRatey_allseeds1[idx].append(rate_base)
			if control == 'o': BaseRateo_allseeds1[idx].append(rate_base)
			if (control == 'y') & (rate_base > 0): SNRy_allseeds1[idx].append(SNR)
			if (control == 'o') & (rate_base > 0): SNRo_allseeds1[idx].append(SNR)
			tv_base = np.around(np.array(SPIKES2[j][(SPIKES2[j]>transient) & (SPIKES2[j]<stimtime)]),3)
			rate_base = (len(tv_base)*1000)/(stimtime-transient) # convert to 1/seconds
			tv = np.around(np.array(SPIKES2[j][(SPIKES2[j]>startslice) & (SPIKES2[j]<endslice)]),3)
			rate = (len(tv)*1000)/(endslice-startslice) # convert to 1/seconds
			SNR = rate/rate_base if (rate_base > 0) else 0
			if control == 'y': Ratesy_allseeds2[idx].append(rate)
			if control == 'o': Rateso_allseeds2[idx].append(rate)
			if control == 'y': BaseRatey_allseeds2[idx].append(rate_base)
			if control == 'o': BaseRateo_allseeds2[idx].append(rate_base)
			if (control == 'y') & (rate_base > 0): SNRy_allseeds2[idx].append(SNR)
			if (control == 'o') & (rate_base > 0): SNRo_allseeds2[idx].append(SNR)

# Initialize dataframe containing stats
df = pd.DataFrame(columns=["Cells", "Metric", "Mean Young", "SD Young", "Mean Old", "SD Old", "t-stat", "p-value", "Cohen's d"])

# All mean rates
MeanRatey = []
MeanRateo = []
MeanBaseRatey = []
MeanBaseRateo = []
PercentActivey = []
PercentActiveo = []
RateCVy = []
RateCVo = []
SNRy = []
SNRo = []
for i in range(0,len(rndinds)):
	rvy = np.mean(Ratesy_allseeds1[i])
	rvo = np.mean(Rateso_allseeds1[i])
	MeanRatey.append(rvy)
	MeanRateo.append(rvo)
	# rvy = np.mean(Ratesy_allseeds2[i])
	# rvo = np.mean(Rateso_allseeds2[i])
	# MeanRatey.append(rvy)
	# MeanRateo.append(rvo)
	rvy = np.mean(BaseRatey_allseeds1[i])
	rvo = np.mean(BaseRateo_allseeds1[i])
	MeanBaseRatey.append(rvy)
	MeanBaseRateo.append(rvo)
	# rvy = np.mean(BaseRatey_allseeds2[i])
	# rvo = np.mean(BaseRateo_allseeds2[i])
	# MeanBaseRatey.append(rvy)
	# MeanBaseRateo.append(rvo)
	rvy = np.mean(SNRy_allseeds1[i])
	rvo = np.mean(SNRo_allseeds1[i])
	SNRy.append(rvy)
	SNRo.append(rvo)
	# rvy = np.mean(SNRy_allseeds2[i])
	# rvo = np.mean(SNRo_allseeds2[i])
	# SNRy.append(rvy)
	# SNRo.append(rvo)
	rvy = np.array(Ratesy_allseeds1[i])
	rvo = np.array(Rateso_allseeds1[i])
	PercentActivey.append((len(rvy[rvy>0])/N_HL5PN)*100)
	PercentActiveo.append((len(rvo[rvo>0])/N_HL5PN)*100)
	# rvy = np.array(Ratesy_allseeds2[i])
	# rvo = np.array(Rateso_allseeds2[i])
	# PercentActivey.append((len(rvy[rvy>0])/N_HL5PN)*100)
	# PercentActiveo.append((len(rvo[rvo>0])/N_HL5PN)*100)
	rvy = np.std(Ratesy_allseeds1[i])/np.mean(Ratesy_allseeds1[i])
	rvo = np.std(Rateso_allseeds1[i])/np.mean(Rateso_allseeds1[i])
	RateCVy.append(rvy)
	RateCVo.append(rvo)
	# rvy = np.std(Ratesy_allseeds2[i])/np.mean(Ratesy_allseeds2[i])
	# rvo = np.std(Rateso_allseeds2[i])/np.mean(Rateso_allseeds2[i])
	# RateCVy.append(rvy)
	# RateCVo.append(rvo)


# Use this code for bootstrapping instead
bsMeans = False
if bsMeans:
	x = bs.bootstrap(np.array(MeanRatey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanratey = x.value
	lowerratey = x.lower_bound
	upperratey = x.upper_bound
	x = bs.bootstrap(np.array(MeanRateo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanrateo = x.value
	lowerrateo = x.lower_bound
	upperrateo = x.upper_bound
	x = bs.bootstrap(np.array(MeanBaseRatey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanbaseratey = x.value
	lowerbaseratey = x.lower_bound
	upperbaseratey = x.upper_bound
	x = bs.bootstrap(np.array(MeanBaseRateo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanbaserateo = x.value
	lowerbaserateo = x.lower_bound
	upperbaserateo = x.upper_bound
	x = bs.bootstrap(np.array(SNRy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRy = x.value
	lowerSNRy = x.lower_bound
	upperSNRy = x.upper_bound
	x = bs.bootstrap(np.array(SNRo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRo = x.value
	lowerSNRo = x.lower_bound
	upperSNRo = x.upper_bound
	x = bs.bootstrap(np.array(PercentActivey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactivey = x.value
	loweractivey = x.lower_bound
	upperactivey = x.upper_bound
	x = bs.bootstrap(np.array(PercentActiveo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactiveo = x.value
	loweractiveo = x.lower_bound
	upperactiveo = x.upper_bound
	x = bs.bootstrap(np.array(RateCVy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVy = x.value
	lowerCVy = x.lower_bound
	upperCVy = x.upper_bound
	x = bs.bootstrap(np.array(RateCVo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVo = x.value
	lowerCVo = x.lower_bound
	upperCVo = x.upper_bound
else:
	meanratey = np.mean(MeanRatey)
	lowerratey = np.std(MeanRatey)
	upperratey = np.std(MeanRatey)
	meanrateo = np.mean(MeanRateo)
	lowerrateo = np.std(MeanRateo)
	upperrateo = np.std(MeanRateo)
	meanbaseratey = np.mean(MeanBaseRatey)
	lowerbaseratey = np.std(MeanBaseRatey)
	upperbaseratey = np.std(MeanBaseRatey)
	meanbaserateo = np.mean(MeanBaseRateo)
	lowerbaserateo = np.std(MeanBaseRateo)
	upperbaserateo = np.std(MeanBaseRateo)
	meanSNRy = np.mean(SNRy)
	lowerSNRy = np.std(SNRy)
	upperSNRy = np.std(SNRy)
	meanSNRo = np.mean(SNRo)
	lowerSNRo = np.std(SNRo)
	upperSNRo = np.std(SNRo)
	meanactivey = np.mean(PercentActivey)
	loweractivey = np.std(PercentActivey)
	upperactivey = np.std(PercentActivey)
	meanactiveo = np.mean(PercentActiveo)
	loweractiveo = np.std(PercentActiveo)
	upperactiveo = np.std(PercentActiveo)
	meanCVy = np.mean(RateCVy)
	lowerCVy = np.std(RateCVy)
	upperCVy = np.std(RateCVy)
	meanCVo = np.mean(RateCVo)
	lowerCVo = np.std(RateCVo)
	upperCVo = np.std(RateCVo)

df = pd.DataFrame(columns=["Cells", "Metric", "Mean Young", "SD Young", "Mean Old", "SD Old", "t-stat", "p-value", "Cohen's d"])

lay0 = abs(meanbaseratey-lowerbaseratey)
uay0 = abs(meanbaseratey-upperbaseratey)
lao0 = abs(meanbaserateo-lowerbaserateo)
uao0 = abs(meanbaserateo-upperbaserateo)

lay = abs(meanratey-lowerratey)
uay = abs(meanratey-upperratey)
lao = abs(meanrateo-lowerrateo)
uao = abs(meanrateo-upperrateo)

tstat_S0, pval_S0 = st.ttest_rel(MeanBaseRatey,MeanRatey)
tstat_S1, pval_S1 = st.ttest_rel(MeanBaseRateo,MeanRateo)
tstat_S2, pval_S2 = st.ttest_rel(MeanBaseRatey,MeanBaseRateo)
tstat_S3, pval_S3 = st.ttest_rel(MeanRatey,MeanRateo)
cd0 = cohen_d(MeanBaseRatey,MeanRatey)
cd1 = cohen_d(MeanBaseRateo,MeanRateo)
cd2 = cohen_d(MeanBaseRatey,MeanBaseRateo)
cd3 = cohen_d(MeanRatey,MeanRateo)

df2 = pd.DataFrame(columns=["Cells",
			"Mean Base Young",
			"SD Base Young",
			"Mean Response Young",
			"SD Response Young",
			"Mean Base Old",
			"SD Base Old",
			"Mean Response Old",
			"SD Response Old",
			"t-stat (base-resp Y vs Y)",
			"p-value (base-resp Y vs Y)",
			"Cohen's d (base-resp Y vs Y)",
			"t-stat (base-resp O vs O)",
			"p-value (base-resp O vs O)",
			"Cohen's d (base-resp O vs O)",
			"t-stat (base-base Y vs O)",
			"p-value (base-base Y vs O)",
			"Cohen's d (base-base Y vs O)",
			"t-stat (resp-resp Y vs O)",
			"p-value (resp-resp Y vs O)",
			"Cohen's d (resp-resp Y vs O)"])

df2 = df2.append({"Cells" : 'All',
			"Mean Base Young" : meanbaseratey,
			"SD Base Young" : [lay0,lay0] if bsMeans else lowerbaseratey,
			"Mean Response Young" : meanratey,
			"SD Response Young" : [lay,lay] if bsMeans else lowerratey,
			"Mean Base Old" : meanbaserateo,
			"SD Base Old" : [lao0,lao0] if bsMeans else lowerbaserateo,
			"Mean Response Old" : meanrateo,
			"SD Response Old" : [lao,lao] if bsMeans else lowerrateo,
			"t-stat (base-resp Y vs Y)" : tstat_S0,
			"p-value (base-resp Y vs Y)" : pval_S0,
			"Cohen's d (base-resp Y vs Y)" : cd0,
			"t-stat (base-resp O vs O)" : tstat_S1,
			"p-value (base-resp O vs O)" : pval_S1,
			"Cohen's d (base-resp O vs O)" : cd1,
			"t-stat (base-base Y vs O)" : tstat_S2,
			"p-value (base-base Y vs O)" : pval_S2,
			"Cohen's d (base-base Y vs O)" : cd2,
			"t-stat (resp-resp Y vs O)" : tstat_S3,
			"p-value (resp-resp Y vs O)" : pval_S3,
			"Cohen's d (resp-resp Y vs O)" : cd3},
			ignore_index = True)

df2.to_csv('figs_tuning/stats_BaseVsResponseRates_'+str(totalstim)+'ms.csv')

x = [0.1,0.9,2.1,2.9]
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanbaseratey,meanbaserateo,meanratey,meanrateo], yerr=[[lay0,lao0],[uay0,uao0],[lay,lao],[uay,uao]] if bsMeans else [lowerbaseratey,lowerbaserateo,lowerratey,lowerrateo], color=['darkgray','lightcoral','darkgray','lightcoral'], width=0.8,tick_label=['Young','Old','Young','Old'],linewidth=1,ecolor='black',error_kw={'elinewidth':4,'markeredgewidth':4},capsize=12,edgecolor='k')
# if pval_S0 < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', [x[i] for i in [0, 2]], [meanratey+2,meanratey+2], yerr=[uay0,uay] if bsMeans else [lowerbaseratey,lowerratey])
# elif pval_S0 < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', [x[i] for i in [0, 2]], [meanratey+2,meanratey+2], yerr=[uay0,uay] if bsMeans else [lowerbaseratey,lowerratey])
# elif pval_S0 < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', [x[i] for i in [0, 2]], [meanratey+2,meanratey+2], yerr=[uay0,uay] if bsMeans else [lowerbaseratey,lowerratey])
# if pval_S1 < 0.001:
# 	barplot_annotate_brackets(0, 1, '***', [x[i] for i in [1, 3]], [meanratey+3.5,meanratey+3.5], yerr=[uao0,uao] if bsMeans else [lowerbaserateo,lowerrateo])
# elif pval_S1 < 0.01:
# 	barplot_annotate_brackets(0, 1, '**', [x[i] for i in [1, 3]], [meanratey+3.5,meanratey+3.5], yerr=[uao0,uao] if bsMeans else [lowerbaserateo,lowerrateo])
# elif pval_S1 < 0.05:
# 	barplot_annotate_brackets(0, 1, '*', [x[i] for i in [1, 3]], [meanratey+3.5,meanratey+3.5], yerr=[uao0,uao] if bsMeans else [lowerbaserateo,lowerrateo])
if pval_S2 < 0.001:
	barplot_annotate_brackets(0, 1, '***', [x[i] for i in [0, 1]], [meanbaseratey,meanbaserateo], yerr=[uay0,uao0] if bsMeans else [lowerbaseratey,lowerbaserateo])
elif pval_S2 < 0.01:
	barplot_annotate_brackets(0, 1, '**', [x[i] for i in [0, 1]], [meanbaseratey,meanbaserateo], yerr=[uay0,uao0] if bsMeans else [lowerbaseratey,lowerbaserateo])
elif pval_S2 < 0.05:
	barplot_annotate_brackets(0, 1, '*', [x[i] for i in [0, 1]], [meanbaseratey,meanbaserateo], yerr=[uay0,uao0] if bsMeans else [lowerbaseratey,lowerbaserateo])
if pval_S3 < 0.001:
	barplot_annotate_brackets(0, 1, '***', [x[i] for i in [2, 3]], [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S3 < 0.01:
	barplot_annotate_brackets(0, 1, '**', [x[i] for i in [2, 3]], [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S3 < 0.05:
	barplot_annotate_brackets(0, 1, '*', [x[i] for i in [2, 3]], [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
ax1.set_ylabel('Mean Pyr Rate (Hz)')
ax1.set_xlim(-0.6,3.6)
ax1.set_xticks([0.5,2.5])
ax1.set_xticklabels(['Pre-stimulus','Post-stimulus'])
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/MeanBaseVsResponseRates_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)



lay = abs(meanactivey-loweractivey)
uay = abs(meanactivey-upperactivey)
lao = abs(meanactiveo-loweractiveo)
uao = abs(meanactiveo-upperactiveo)

tstat_S, pval_S = st.ttest_rel(PercentActivey,PercentActiveo)
cd = cohen_d(PercentActivey,PercentActiveo)

df = df.append({"Cells" : 'All',
			"Metric" : '% Active',
			"Mean Young" : meanactivey,
			"SD Young" : [lay,lay] if bsMeans else loweractivey,
			"Mean Old" : meanactiveo,
			"SD Old" : [lao,lao] if bsMeans else loweractiveo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanactivey,meanactiveo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [loweractivey,loweractiveo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('% Silent p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
ax1.set_ylabel('Pyr % Active During Response')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/PercentActive_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

lay = abs(meanratey-lowerratey)
uay = abs(meanratey-upperratey)
lao = abs(meanrateo-lowerrateo)
uao = abs(meanrateo-upperrateo)

tstat_S, pval_S = st.ttest_rel(MeanRatey,MeanRateo)
cd = cohen_d(MeanRatey,MeanRateo)

df = df.append({"Cells" : 'All',
			"Metric" : 'Response Rate',
			"Mean Young" : meanratey,
			"SD Young" : [lay,lay] if bsMeans else lowerratey,
			"Mean Old" : meanrateo,
			"SD Old" : [lao,lao] if bsMeans else lowerrateo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanratey,meanrateo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerratey,lowerrateo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('Mean Rate p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
ax1.set_ylabel('Mean Pyr Response Rate (Hz)')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/MeanRate_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)


lay = abs(meanSNRy-lowerSNRy)
uay = abs(meanSNRy-upperSNRy)
lao = abs(meanSNRo-lowerSNRo)
uao = abs(meanSNRo-upperSNRo)

tstat_S, pval_S = st.ttest_rel(SNRy,SNRo)
cd = cohen_d(SNRy,SNRo)

df = df.append({"Cells" : 'All',
			"Metric" : 'SNR',
			"Mean Young" : meanSNRy,
			"SD Young" : [lay,lay] if bsMeans else lowerSNRy,
			"Mean Old" : meanSNRo,
			"SD Old" : [lao,lao] if bsMeans else lowerSNRo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [-0.6,1.6]
fig = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanSNRy,meanSNRo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerSNRy,lowerSNRo], color=['darkgray','lightcoral'], width=2.2,tick_label=['Younger','Older'],linewidth=1,ecolor='black',error_kw={'elinewidth':4,'markeredgewidth':4},capsize=12,edgecolor='k')
print('SNR p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', [x[i] for i in [0, 1]], [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', [x[i] for i in [0, 1]], [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', [x[i] for i in [0, 1]], [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
ax1.set_ylabel('Pyr SNR')
ax1.set_xlim(-2,3)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/SNR_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)


lay = abs(meanCVy-lowerCVy)
uay = abs(meanCVy-upperCVy)
lao = abs(meanCVo-lowerCVo)
uao = abs(meanCVo-upperCVo)

tstat_S, pval_S = st.ttest_rel(RateCVy,RateCVo)
cd = cohen_d(RateCVy,RateCVo)

df = df.append({"Cells" : 'All',
			"Metric" : 'Response Rate CV',
			"Mean Young" : meanCVy,
			"SD Young" : [lay,lay] if bsMeans else lowerCVy,
			"Mean Old" : meanCVo,
			"SD Old" : [lao,lao] if bsMeans else lowerCVo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanCVy,meanCVo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerCVy,lowerCVo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('CV p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
ax1.set_ylabel('Response Rate CV')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/RateCV_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

# Stimulated mean rate plots
MeanRatey = []
MeanRateo = []
PercentActivey = []
PercentActiveo = []
RateCVy = []
RateCVo = []
SNRy = []
SNRo = []
for i in range(0,len(rndinds)):
	rvy = np.mean(Ratesy_allseeds1[i][:N_HL5PNso])
	rvo = np.mean(Rateso_allseeds1[i][:N_HL5PNso])
	MeanRatey.append(rvy)
	MeanRateo.append(rvo)
	rvy = np.mean(Ratesy_allseeds2[i][:N_HL5PNso])
	rvo = np.mean(Rateso_allseeds2[i][:N_HL5PNso])
	MeanRatey.append(rvy)
	MeanRateo.append(rvo)
	rvy = np.mean(SNRy_allseeds1[i][:N_HL5PNso])
	rvo = np.mean(SNRo_allseeds1[i][:N_HL5PNso])
	SNRy.append(rvy)
	SNRo.append(rvo)
	rvy = np.mean(SNRy_allseeds2[i][:N_HL5PNso])
	rvo = np.mean(SNRo_allseeds2[i][:N_HL5PNso])
	SNRy.append(rvy)
	SNRo.append(rvo)
	rvy = np.array(Ratesy_allseeds1[i][:N_HL5PNso])
	rvo = np.array(Rateso_allseeds1[i][:N_HL5PNso])
	PercentActivey.append((len(rvy[rvy>0])/N_HL5PNso)*100)
	PercentActiveo.append((len(rvo[rvo>0])/N_HL5PNso)*100)
	rvy = np.array(Ratesy_allseeds2[i][:N_HL5PNso])
	rvo = np.array(Rateso_allseeds2[i][:N_HL5PNso])
	PercentActivey.append((len(rvy[rvy>0])/N_HL5PNso)*100)
	PercentActiveo.append((len(rvo[rvo>0])/N_HL5PNso)*100)
	rvy = np.std(Ratesy_allseeds1[i][:N_HL5PNso])/np.mean(Ratesy_allseeds1[i][:N_HL5PNso])
	rvo = np.std(Rateso_allseeds1[i][:N_HL5PNso])/np.mean(Rateso_allseeds1[i][:N_HL5PNso])
	RateCVy.append(rvy)
	RateCVo.append(rvo)
	rvy = np.std(Ratesy_allseeds2[i][:N_HL5PNso])/np.mean(Ratesy_allseeds2[i][:N_HL5PNso])
	rvo = np.std(Rateso_allseeds2[i][:N_HL5PNso])/np.mean(Rateso_allseeds2[i][:N_HL5PNso])
	RateCVy.append(rvy)
	RateCVo.append(rvo)

# Use this code for bootstrapping instead
if bsMeans:
	x = bs.bootstrap(np.array(MeanRatey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanratey = x.value
	lowerratey = x.lower_bound
	upperratey = x.upper_bound
	x = bs.bootstrap(np.array(MeanRateo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanrateo = x.value
	lowerrateo = x.lower_bound
	upperrateo = x.upper_bound
	x = bs.bootstrap(np.array(SNRy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRy = x.value
	lowerSNRy = x.lower_bound
	upperSNRy = x.upper_bound
	x = bs.bootstrap(np.array(SNRo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRo = x.value
	lowerSNRo = x.lower_bound
	upperSNRo = x.upper_bound
	x = bs.bootstrap(np.array(PercentActivey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactivey = x.value
	loweractivey = x.lower_bound
	upperactivey = x.upper_bound
	x = bs.bootstrap(np.array(PercentActiveo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactiveo = x.value
	loweractiveo = x.lower_bound
	upperactiveo = x.upper_bound
	x = bs.bootstrap(np.array(RateCVy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVy = x.value
	lowerCVy = x.lower_bound
	upperCVy = x.upper_bound
	x = bs.bootstrap(np.array(RateCVo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVo = x.value
	lowerCVo = x.lower_bound
	upperCVo = x.upper_bound
else:
	meanratey = np.mean(MeanRatey)
	lowerratey = np.std(MeanRatey)
	upperratey = np.std(MeanRatey)
	meanrateo = np.mean(MeanRateo)
	lowerrateo = np.std(MeanRateo)
	upperrateo = np.std(MeanRateo)
	meanSNRy = np.mean(SNRy)
	lowerSNRy = np.std(SNRy)
	upperSNRy = np.std(SNRy)
	meanSNRo = np.mean(SNRo)
	lowerSNRo = np.std(SNRo)
	upperSNRo = np.std(SNRo)
	meanactivey = np.mean(PercentActivey)
	loweractivey = np.std(PercentActivey)
	upperactivey = np.std(PercentActivey)
	meanactiveo = np.mean(PercentActiveo)
	loweractiveo = np.std(PercentActiveo)
	upperactiveo = np.std(PercentActiveo)
	meanCVy = np.mean(RateCVy)
	lowerCVy = np.std(RateCVy)
	upperCVy = np.std(RateCVy)
	meanCVo = np.mean(RateCVo)
	lowerCVo = np.std(RateCVo)
	upperCVo = np.std(RateCVo)

lay = abs(meanactivey-loweractivey)
uay = abs(meanactivey-upperactivey)
lao = abs(meanactiveo-loweractiveo)
uao = abs(meanactiveo-upperactiveo)

tstat_S, pval_S = st.ttest_rel(PercentActivey,PercentActiveo)
cd = cohen_d(PercentActivey,PercentActiveo)

df = df.append({"Cells" : 'Stimulated',
			"Metric" : '% Active',
			"Mean Young" : meanactivey,
			"SD Young" : [lay,lay] if bsMeans else loweractivey,
			"Mean Old" : meanactiveo,
			"SD Old" : [lao,lao] if bsMeans else loweractiveo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanactivey,meanactiveo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [loweractivey,loweractiveo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('% Silent p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
ax1.set_ylabel('Stimulated Pyr % Active During Response')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/PercentActive_'+str(totalstim)+'ms_StimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

lay = abs(meanratey-lowerratey)
uay = abs(meanratey-upperratey)
lao = abs(meanrateo-lowerrateo)
uao = abs(meanrateo-upperrateo)

tstat_S, pval_S = st.ttest_rel(MeanRatey,MeanRateo)
cd = cohen_d(MeanRatey,MeanRateo)

df = df.append({"Cells" : 'Stimulated',
			"Metric" : 'Response Rate',
			"Mean Young" : meanratey,
			"SD Young" : [lay,lay] if bsMeans else lowerratey,
			"Mean Old" : meanrateo,
			"SD Old" : [lao,lao] if bsMeans else lowerrateo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanratey,meanrateo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerratey,lowerrateo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('Mean Rate p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
ax1.set_ylabel('Mean Stimulated Pyr Response Rate (Hz)')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/MeanRate_'+str(totalstim)+'ms_StimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)


lay = abs(meanSNRy-lowerSNRy)
uay = abs(meanSNRy-upperSNRy)
lao = abs(meanSNRo-lowerSNRo)
uao = abs(meanSNRo-upperSNRo)

tstat_S, pval_S = st.ttest_rel(SNRy,SNRo)
cd = cohen_d(SNRy,SNRo)

df = df.append({"Cells" : 'Stimulated',
			"Metric" : 'SNR',
			"Mean Young" : meanSNRy,
			"SD Young" : [lay,lay] if bsMeans else lowerSNRy,
			"Mean Old" : meanSNRo,
			"SD Old" : [lao,lao] if bsMeans else lowerSNRo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanSNRy,meanSNRo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerSNRy,lowerSNRo], color=['dimgray','r'], width=0.8,tick_label=['Younger','Older'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('SNR p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
ax1.set_ylabel('Stimulated Pyr SNR')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/SNR_'+str(totalstim)+'ms_StimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)


lay = abs(meanCVy-lowerCVy)
uay = abs(meanCVy-upperCVy)
lao = abs(meanCVo-lowerCVo)
uao = abs(meanCVo-upperCVo)

tstat_S, pval_S = st.ttest_rel(RateCVy,RateCVo)
cd = cohen_d(RateCVy,RateCVo)

df = df.append({"Cells" : 'Stimulated',
			"Metric" : 'Response Rate CV',
			"Mean Young" : meanCVy,
			"SD Young" : [lay,lay] if bsMeans else lowerCVy,
			"Mean Old" : meanCVo,
			"SD Old" : [lao,lao] if bsMeans else lowerCVo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanCVy,meanCVo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerCVy,lowerCVo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('CV p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
ax1.set_ylabel('Stimulated Pyr Response Rate CV')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/RateCV_'+str(totalstim)+'ms_StimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

# Untimulated mean rate plots
MeanRatey = []
MeanRateo = []
PercentActivey = []
PercentActiveo = []
RateCVy = []
RateCVo = []
SNRy = []
SNRo = []
for i in range(0,len(rndinds)):
	rvy = np.mean(Ratesy_allseeds1[i][N_HL5PNso:])
	rvo = np.mean(Rateso_allseeds1[i][N_HL5PNso:])
	MeanRatey.append(rvy)
	MeanRateo.append(rvo)
	rvy = np.mean(Ratesy_allseeds2[i][N_HL5PNso:])
	rvo = np.mean(Rateso_allseeds2[i][N_HL5PNso:])
	MeanRatey.append(rvy)
	MeanRateo.append(rvo)
	rvy = np.mean(SNRy_allseeds1[i][N_HL5PNso:])
	rvo = np.mean(SNRo_allseeds1[i][N_HL5PNso:])
	SNRy.append(rvy)
	SNRo.append(rvo)
	rvy = np.mean(SNRy_allseeds2[i][N_HL5PNso:])
	rvo = np.mean(SNRo_allseeds2[i][N_HL5PNso:])
	SNRy.append(rvy)
	SNRo.append(rvo)
	rvy = np.array(Ratesy_allseeds1[i][N_HL5PNso:])
	rvo = np.array(Rateso_allseeds1[i][N_HL5PNso:])
	PercentActivey.append((len(rvy[rvy>0])/(N_HL5PN-N_HL5PNso))*100)
	PercentActiveo.append((len(rvo[rvo>0])/(N_HL5PN-N_HL5PNso))*100)
	rvy = np.array(Ratesy_allseeds2[i][N_HL5PNso:])
	rvo = np.array(Rateso_allseeds2[i][N_HL5PNso:])
	PercentActivey.append((len(rvy[rvy>0])/(N_HL5PN-N_HL5PNso))*100)
	PercentActiveo.append((len(rvo[rvo>0])/(N_HL5PN-N_HL5PNso))*100)
	rvy = np.std(Ratesy_allseeds1[i][N_HL5PNso:])/np.mean(Ratesy_allseeds1[i][N_HL5PNso:])
	rvo = np.std(Rateso_allseeds1[i][N_HL5PNso:])/np.mean(Rateso_allseeds1[i][N_HL5PNso:])
	RateCVy.append(rvy)
	RateCVo.append(rvo)
	rvy = np.std(Ratesy_allseeds2[i][N_HL5PNso:])/np.mean(Ratesy_allseeds2[i][N_HL5PNso:])
	rvo = np.std(Rateso_allseeds2[i][N_HL5PNso:])/np.mean(Rateso_allseeds2[i][N_HL5PNso:])
	RateCVy.append(rvy)
	RateCVo.append(rvo)

# Use this code for bootstrapping instead
if bsMeans:
	x = bs.bootstrap(np.array(MeanRatey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanratey = x.value
	lowerratey = x.lower_bound
	upperratey = x.upper_bound
	x = bs.bootstrap(np.array(MeanRateo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanrateo = x.value
	lowerrateo = x.lower_bound
	upperrateo = x.upper_bound
	x = bs.bootstrap(np.array(SNRy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRy = x.value
	lowerSNRy = x.lower_bound
	upperSNRy = x.upper_bound
	x = bs.bootstrap(np.array(SNRo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanSNRo = x.value
	lowerSNRo = x.lower_bound
	upperSNRo = x.upper_bound
	x = bs.bootstrap(np.array(PercentActivey), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactivey = x.value
	loweractivey = x.lower_bound
	upperactivey = x.upper_bound
	x = bs.bootstrap(np.array(PercentActiveo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanactiveo = x.value
	loweractiveo = x.lower_bound
	upperactiveo = x.upper_bound
	x = bs.bootstrap(np.array(RateCVy), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVy = x.value
	lowerCVy = x.lower_bound
	upperCVy = x.upper_bound
	x = bs.bootstrap(np.array(RateCVo), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	meanCVo = x.value
	lowerCVo = x.lower_bound
	upperCVo = x.upper_bound
else:
	meanratey = np.mean(MeanRatey)
	lowerratey = np.std(MeanRatey)
	upperratey = np.std(MeanRatey)
	meanrateo = np.mean(MeanRateo)
	lowerrateo = np.std(MeanRateo)
	upperrateo = np.std(MeanRateo)
	meanSNRy = np.mean(SNRy)
	lowerSNRy = np.std(SNRy)
	upperSNRy = np.std(SNRy)
	meanSNRo = np.mean(SNRo)
	lowerSNRo = np.std(SNRo)
	upperSNRo = np.std(SNRo)
	meanactivey = np.mean(PercentActivey)
	loweractivey = np.std(PercentActivey)
	upperactivey = np.std(PercentActivey)
	meanactiveo = np.mean(PercentActiveo)
	loweractiveo = np.std(PercentActiveo)
	upperactiveo = np.std(PercentActiveo)
	meanCVy = np.mean(RateCVy)
	lowerCVy = np.std(RateCVy)
	upperCVy = np.std(RateCVy)
	meanCVo = np.mean(RateCVo)
	lowerCVo = np.std(RateCVo)
	upperCVo = np.std(RateCVo)

lay = abs(meanactivey-loweractivey)
uay = abs(meanactivey-upperactivey)
lao = abs(meanactiveo-loweractiveo)
uao = abs(meanactiveo-upperactiveo)

tstat_S, pval_S = st.ttest_rel(PercentActivey,PercentActiveo)
cd = cohen_d(PercentActivey,PercentActiveo)

df = df.append({"Cells" : 'Recurrent',
			"Metric" : '% Active',
			"Mean Young" : meanactivey,
			"SD Young" : [lay,lay] if bsMeans else loweractivey,
			"Mean Old" : meanactiveo,
			"SD Old" : [lao,lao] if bsMeans else loweractiveo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanactivey,meanactiveo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [loweractivey,loweractiveo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('% Silent p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanactivey,meanactiveo], yerr=[uay,uao]if bsMeans else [loweractivey,loweractiveo])
ax1.set_ylabel('Recurrent Pyr % Active During Response')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/PercentActive_'+str(totalstim)+'ms_UnstimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

lay = abs(meanratey-lowerratey)
uay = abs(meanratey-upperratey)
lao = abs(meanrateo-lowerrateo)
uao = abs(meanrateo-upperrateo)

tstat_S, pval_S = st.ttest_rel(MeanRatey,MeanRateo)
cd = cohen_d(MeanRatey,MeanRateo)

df = df.append({"Cells" : 'Recurrent',
			"Metric" : 'Response Rate',
			"Mean Young" : meanratey,
			"SD Young" : [lay,lay] if bsMeans else lowerratey,
			"Mean Old" : meanrateo,
			"SD Old" : [lao,lao] if bsMeans else lowerrateo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanratey,meanrateo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerratey,lowerrateo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('Mean Rate p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanratey,meanrateo], yerr=[uay,uao] if bsMeans else [lowerratey,lowerrateo])
ax1.set_ylabel('Mean Recurrent Pyr Response Rate (Hz)')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/MeanRate_'+str(totalstim)+'ms_UnstimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)


lay = abs(meanSNRy-lowerSNRy)
uay = abs(meanSNRy-upperSNRy)
lao = abs(meanSNRo-lowerSNRo)
uao = abs(meanSNRo-upperSNRo)

tstat_S, pval_S = st.ttest_rel(SNRy,SNRo)
cd = cohen_d(SNRy,SNRo)

df = df.append({"Cells" : 'Recurrent',
			"Metric" : 'SNR',
			"Mean Young" : meanSNRy,
			"SD Young" : [lay,lay] if bsMeans else lowerSNRy,
			"Mean Old" : meanSNRo,
			"SD Old" : [lao,lao] if bsMeans else lowerSNRo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanSNRy,meanSNRo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerSNRy,lowerSNRo], color=['dimgray','r'], width=0.8,tick_label=['Younger','Older'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('SNR p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanSNRy,meanSNRo], yerr=[uay,uao] if bsMeans else [lowerSNRy,lowerSNRo])
ax1.set_ylabel('Recurrent Pyr SNR')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/SNR_'+str(totalstim)+'ms_UnstimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)



lay = abs(meanCVy-lowerCVy)
uay = abs(meanCVy-upperCVy)
lao = abs(meanCVo-lowerCVo)
uao = abs(meanCVo-upperCVo)

tstat_S, pval_S = st.ttest_rel(RateCVy,RateCVo)
cd = cohen_d(RateCVy,RateCVo)

df = df.append({"Cells" : 'Recurrent',
			"Metric" : 'Response Rate CV',
			"Mean Young" : meanCVy,
			"SD Young" : [lay,lay] if bsMeans else lowerCVy,
			"Mean Old" : meanCVo,
			"SD Old" : [lao,lao] if bsMeans else lowerCVo,
			"t-stat" : tstat_S,
			"p-value" : pval_S,
			"Cohen's d" : cd},
			ignore_index = True)

x = [0,1]
fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(111)
ax1.bar(x, [meanCVy,meanCVo], yerr=[[lay,lao],[uay,uao]] if bsMeans else [lowerCVy,lowerCVo], color=['dimgray','r'], width=0.8,tick_label=['Young','Old'],linewidth=5,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
print('CV p-value='+str(pval_S))
if pval_S < 0.001:
	barplot_annotate_brackets(0, 1, '***', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.01:
	barplot_annotate_brackets(0, 1, '**', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
elif pval_S < 0.05:
	barplot_annotate_brackets(0, 1, '*', x, [meanCVy,meanCVo], yerr=[uay,uao] if bsMeans else [lowerCVy,lowerCVo])
ax1.set_ylabel('Unstimulated Pyr Response Rate CV')
ax1.set_xlim(-0.6,1.6)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/RateCV_'+str(totalstim)+'ms_UnstimulatedCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

# Save stats dataframe to csv
df.to_csv('figs_tuning/stats_'+str(totalstim)+'ms.csv')

# Stimulated mean rates
MeanRatey1 = []
MeanRateo1 = []
MeanRatey2 = []
MeanRateo2 = []
for i in range(0,len(rndinds)):
	rvy = np.mean(Ratesy_allseeds1[i][:N_HL5PNso])
	rvo = np.mean(Rateso_allseeds1[i][:N_HL5PNso])
	MeanRatey1.append(rvy)
	MeanRateo1.append(rvo)
	rvy = np.mean(Ratesy_allseeds2[i][:N_HL5PNso])
	rvo = np.mean(Rateso_allseeds2[i][:N_HL5PNso])
	MeanRatey2.append(rvy)
	MeanRateo2.append(rvo)

x = bs.bootstrap(np.array(MeanRatey1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanratey1 = x.value
x = bs.bootstrap(np.array(MeanRateo1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanrateo1 = x.value
x = bs.bootstrap(np.array(MeanRatey2), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanratey2 = x.value
x = bs.bootstrap(np.array(MeanRateo2), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanrateo2 = x.value

mrys1 = meanratey1
mros1 = meanrateo1
mrys2 = meanratey2
mros2 = meanrateo2

# Unstimulated mean rates
MeanRatey1 = []
MeanRateo1 = []
MeanRatey2 = []
MeanRateo2 = []
for i in range(0,len(rndinds)):
	rvy = np.mean(Ratesy_allseeds1[i][N_HL5PNso:])
	rvo = np.mean(Rateso_allseeds1[i][N_HL5PNso:])
	MeanRatey1.append(rvy)
	MeanRateo1.append(rvo)
	rvy = np.mean(Ratesy_allseeds2[i][N_HL5PNso:])
	rvo = np.mean(Rateso_allseeds2[i][N_HL5PNso:])
	MeanRatey2.append(rvy)
	MeanRateo2.append(rvo)

x = bs.bootstrap(np.array(MeanRatey1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanratey1 = x.value
x = bs.bootstrap(np.array(MeanRateo1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanrateo1 = x.value
x = bs.bootstrap(np.array(MeanRatey2), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanratey2 = x.value
x = bs.bootstrap(np.array(MeanRateo2), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
meanrateo2 = x.value

mryu1 = meanratey1
mrou1 = meanrateo1
mryu2 = meanratey2
mrou2 = meanrateo2

# Plot tuning curves
CI_means_PNy1 = []
CI_lower_PNy1 = []
CI_upper_PNy1 = []
CI_means_PNo1 = []
CI_lower_PNo1 = []
CI_upper_PNo1 = []
CI_means_PNy2 = []
CI_lower_PNy2 = []
CI_upper_PNy2 = []
CI_means_PNo2 = []
CI_lower_PNo2 = []
CI_upper_PNo2 = []

bsMeans = False
if bsMeans:
	for l in range(0,N_HL5PN):
		print('Starting Bootstrapping')
		x = bs.bootstrap(np.transpose(Ratesy_allseeds1)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNy1.append(x.value)
		CI_lower_PNy1.append(x.lower_bound)
		CI_upper_PNy1.append(x.upper_bound)
		x = bs.bootstrap(np.transpose(Rateso_allseeds1)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNo1.append(x.value)
		CI_lower_PNo1.append(x.lower_bound)
		CI_upper_PNo1.append(x.upper_bound)
		x = bs.bootstrap(np.transpose(Ratesy_allseeds2)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNy2.append(x.value)
		CI_lower_PNy2.append(x.lower_bound)
		CI_upper_PNy2.append(x.upper_bound)
		x = bs.bootstrap(np.transpose(Rateso_allseeds2)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNo2.append(x.value)
		CI_lower_PNo2.append(x.lower_bound)
		CI_upper_PNo2.append(x.upper_bound)
else:
	for l in range(0,N_HL5PN):
		CI_means_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l]))
		CI_lower_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l])-np.std(np.transpose(Ratesy_allseeds1)[l]))
		CI_upper_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l])+np.std(np.transpose(Ratesy_allseeds1)[l]))
		x = bs.bootstrap(np.transpose(Rateso_allseeds1)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l]))
		CI_lower_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l])-np.std(np.transpose(Rateso_allseeds1)[l]))
		CI_upper_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l])+np.std(np.transpose(Rateso_allseeds1)[l]))
		x = bs.bootstrap(np.transpose(Ratesy_allseeds2)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNy2.append(np.mean(np.transpose(Ratesy_allseeds2)[l]))
		CI_lower_PNy2.append(np.mean(np.transpose(Ratesy_allseeds2)[l])-np.std(np.transpose(Ratesy_allseeds2)[l]))
		CI_upper_PNy2.append(np.mean(np.transpose(Ratesy_allseeds2)[l])+np.std(np.transpose(Ratesy_allseeds2)[l]))
		x = bs.bootstrap(np.transpose(Rateso_allseeds2)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		CI_means_PNo2.append(np.mean(np.transpose(Rateso_allseeds2)[l]))
		CI_lower_PNo2.append(np.mean(np.transpose(Rateso_allseeds2)[l])-np.std(np.transpose(Rateso_allseeds2)[l]))
		CI_upper_PNo2.append(np.mean(np.transpose(Rateso_allseeds2)[l])+np.std(np.transpose(Rateso_allseeds2)[l]))

gscalery1 = np.percentile(CI_means_PNy1,90)-mryu1
gscalero1 = np.percentile(CI_means_PNo1,90)-mrou1
mu_ind1 = int(N_HL5PNso*synnums[0]/180) # find index with peak input depending on orientation
gscalery2 = np.percentile(CI_means_PNy2,90)-mryu2
gscalero2 = np.percentile(CI_means_PNo2,90)-mrou2
mu_ind2 = int(N_HL5PNso*synnums[1]/180) # find index with peak input depending on orientation
sigma = int(N_HL5PNso*42/180) # Set sigma to 42 degrees (Bensmaia et al 2008) and scale by number of PN neurons

x_values = np.linspace(0, N_HL5PN-1, N_HL5PN)
gfuncy1 = gaussian(x_values, mu_ind1, sigma)*gscalery1+mryu1 # Gaussian vector (y=[0,1]) of length = cellnum
gfuncy1[x_values>N_HL5PNso]=mryu1
gfunco1 = gaussian(x_values, mu_ind1, sigma)*gscalero1+mrou1 # Gaussian vector (y=[0,1]) of length = cellnum
gfunco1[x_values>N_HL5PNso]=mrou1
gfuncy2 = gaussian(x_values, mu_ind2, sigma)*gscalery2+mryu2 # Gaussian vector (y=[0,1]) of length = cellnum
gfuncy2[x_values>N_HL5PNso]=mryu2
gfunco2 = gaussian(x_values, mu_ind2, sigma)*gscalero2+mrou2 # Gaussian vector (y=[0,1]) of length = cellnum
gfunco2[x_values>N_HL5PNso]=mrou2

# Plot Heatmap of Tuning for each Cell
x_values = np.linspace(0, N_HL5PN-1, N_HL5PN)
y_values = np.linspace(0, 180, N_HL5PNso)
sigma = int(N_HL5PNso*42/180) # Set sigma to 42 degrees (Bensmaia et al 2008) and scale by number of PN neurons
all_gfs = []
for y in y_values:
	yin = int(N_HL5PNso*y/180)
	gf = gaussian(x_values,yin,sigma)*45
	gf[x_values>=N_HL5PNso]=0
	all_gfs.append(gf)

fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(11,3.2))
im = axarr.imshow(all_gfs,origin='lower',extent = [0 , N_HL5PN-1, 0 , 180], aspect='auto')
axarr.set_ylabel('Angle ($^\circ$)')
axarr.set_xlabel('Neuron Index')
axarr.set_xlim(0,N_HL5PN)
axarr.set_xticks([0,100,200,300,400,500,600,700])
cbar = fig.colorbar(im)
cbar.set_label(r'$N_{inputs}$')
fig.tight_layout()
fig.savefig('figs_tuning/tuning_schematic.png',dpi=300,transparent=True)

xcells = np.arange(0,N_HL5PN,1)
fig, axarr = plt.subplots(nrows=2,ncols=1,figsize=(9,7),sharex=True,sharey=True)
fig.text(0.02, 0.5, 'Pyr Response Rate (Hz)', ha='center', va='center', rotation='vertical')

# Add background shading to highlight feature subsets
N_PNs = 700
N_PNs_stim = int(N_PNs/2)
NFeats = 100
SubsetFeatures1 = [0,N_PNs] # [Logical, Cell Start Index, Cell End Index] - Only triggers if logical = True
SubsetFeatures2 = [0,N_PNs_stim]
NFeats = 200
SubsetFeatures3 = [int(N_PNs_stim/2)-int(NFeats/2),int(N_PNs_stim/2)+int(NFeats/2)]
NFeats = 100
SubsetFeatures4 = [int(N_PNs_stim/2)-int(NFeats/2),int(N_PNs_stim/2)+int(NFeats/2)]
ymax = np.max([np.max(CI_upper_PNy1),np.max(CI_upper_PNy2),np.max(CI_upper_PNo1),np.max(CI_upper_PNo2)])
lb = [0,0]
ub = [ymax,ymax]
# axarr[0].fill_between(SubsetFeatures1,lb,ub,facecolor='k',alpha=0.1)
# axarr[0].fill_between(SubsetFeatures2,lb,ub,facecolor='k',alpha=0.2)
# axarr[0].fill_between(SubsetFeatures3,lb,ub,facecolor='k',alpha=0.3)
# axarr[0].fill_between(SubsetFeatures4,lb,ub,facecolor='k',alpha=0.4)
# axarr[1].fill_between(SubsetFeatures1,lb,ub,facecolor='k',alpha=0.1)
# axarr[1].fill_between(SubsetFeatures2,lb,ub,facecolor='k',alpha=0.2)
# axarr[1].fill_between(SubsetFeatures3,lb,ub,facecolor='k',alpha=0.3)
# axarr[1].fill_between(SubsetFeatures4,lb,ub,facecolor='k',alpha=0.4)


axarr[0].plot(xcells,CI_means_PNy1,color='k',alpha=1,label=str(synnums[0])+r'$\degree$', linewidth=0.5)
axarr[0].fill_between(xcells,CI_lower_PNy1,CI_upper_PNy1,facecolor='k',alpha=0.5)
axarr[0].plot(xcells,CI_means_PNy2,color='tab:brown',alpha=1,label=str(synnums[1])+r'$\degree$', linewidth=0.5)
axarr[0].fill_between(xcells,CI_lower_PNy2,CI_upper_PNy2,facecolor='tab:brown',alpha=0.3)
axarr[0].legend()
axarr[1].plot(xcells,CI_means_PNo1,color='r',alpha=1,label=str(synnums[0])+r'$\degree$', linewidth=0.5)
axarr[1].fill_between(xcells,CI_lower_PNo1,CI_upper_PNo1,facecolor='r',alpha=0.4)
axarr[1].plot(xcells,CI_means_PNo2,color='tab:orange',alpha=1,label=str(synnums[1])+r'$\degree$', linewidth=0.5)
axarr[1].fill_between(xcells,CI_lower_PNo2,CI_upper_PNo2,facecolor='tab:orange',alpha=0.3)
axarr[1].legend()
# axarr[0].plot(x_values,gfuncy1,color='k',alpha=1,lw=3,ls=':')
# axarr[0].plot(x_values,gfuncy2,color='tab:brown',alpha=1,lw=3,ls=':')
# axarr[1].plot(x_values,gfunco1,color='r',alpha=1,lw=3,ls=':')
# axarr[1].plot(x_values,gfunco2,color='tab:orange',alpha=1,lw=3,ls=':')
axarr[0].set_xlim(0,N_HL5PN)
axarr[0].set_ylim(0,ymax)
ylims=axarr[0].get_ylim()
axarr[0].plot(np.array([N_HL5PNso,N_HL5PNso]),ylims,color='dimgray',alpha=1,lw=2,ls='dashed')
axarr[1].plot(np.array([N_HL5PNso,N_HL5PNso]),ylims,color='dimgray',alpha=1,lw=2,ls='dashed')
axarr[0].set_ylim(0,ymax)
# axarr[1].set_ylabel('Spike Rate (Hz)')

secax = axarr[0].secondary_xaxis('top')
xpos = np.arange(0,N_HL5PNso+1,N_HL5PNso/8)
rot = 180/8
secax.set_xticks(xpos)
secax.set_xticklabels([u'\u2014']*len(xpos),fontsize=35,verticalalignment='bottom')
for i, t in enumerate(secax.get_xticklabels()):
	t.set_rotation(i*rot)

axarr[1].set_xlim(0,N_HL5PN)
axarr[1].set_xlabel('Neuron Index')
fig.tight_layout()
fig.savefig('figs_tuning/tuningcurve_'+str(totalstim)+'ms_allcells_Compare.png',dpi=300,transparent=True)


xcells = np.arange(0,N_HL5PN,1)
fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(10,5),sharex=True,sharey=True)

# Add background shading to highlight feature subsets
ymax = np.max(CI_upper_PNy1)
lb = [0,0]
ub = [ymax,ymax]

axarr.scatter(xcells,CI_means_PNy1,c='k',alpha=0.25,label=str(synnums[0])+r'$\degree$', linewidths=1, edgecolors='none')
# axarr.plot(x_values,gfuncy1,color='k',alpha=1,lw=3,ls=':')
# axarr.fill_between(xcells,CI_lower_PNy1,CI_upper_PNy1,facecolor='k',alpha=0.5)
axarr.set_xlim(0,N_HL5PN)
axarr.set_ylim(0,ymax)
ylims=axarr.get_ylim()
axarr.plot(np.array([N_HL5PNso,N_HL5PNso]),ylims,color='dimgray',alpha=1,lw=2,ls='dashed')
axarr.set_ylim(0,ymax)
axarr.set_ylabel('Pyr Response Rate (Hz)')

# axins = inset_axes(axarr, width="40%", height="40%",loc=1)
# axins.plot(x_values,gfuncy1,color='k',alpha=1,lw=3,ls=':')
# axins.fill_between(xcells[0:N_HL5PNso],CI_lower_PNy1[0:N_HL5PNso],CI_upper_PNy1[0:N_HL5PNso],facecolor='k',alpha=0.5)
# axins.set_ylim(0,ymax)

# secax = axarr.secondary_xaxis('top')
xpos = np.arange(0,N_HL5PNso+1,N_HL5PNso/8)
rot = 180/8
# secax.set_xticks(xpos)
# secax.set_xticklabels([u'\u2014']*len(xpos),fontsize=35,verticalalignment='bottom')
# for i, t in enumerate(secax.get_xticklabels()):
# 	t.set_rotation(i*rot)

# axins.set_xticks(xpos)
# axins.set_xticklabels([u'\u2014']*len(xpos),fontsize=35)
# for i, t in enumerate(axins.get_xticklabels()):
# 	t.set_rotation(i*rot)
# axins.set_xlim(5,N_HL5PNso-5)

axarr.set_xlim(0,N_HL5PN)
axarr.set_xlabel('Neuron Index')
fig.tight_layout()
fig.savefig('figs_tuning/tuningcurve_scatter_YoungOnlyA1_'+str(totalstim)+'ms_allcells_Compare.png',dpi=300,transparent=True)



# Cross-correlations
xpos = np.arange(-N_HL5PN+1,N_HL5PN,1)
AllMeans = [CI_means_PNy1,CI_means_PNy2,CI_means_PNo1,CI_means_PNo2]
labels = ['YA1','YA2','OA1','OA2']
fig, axarr = plt.subplots(nrows=len(AllMeans),ncols=len(AllMeans),figsize=(14,14),sharex=True,sharey=True)
i = 0
for a in AllMeans:
	j = 0
	for b in AllMeans:
		# Normalize
		a = (a - np.mean(a)) / (np.std(a) * len(a))
		b = (b - np.mean(b)) / (np.std(b))
		c = np.correlate(a, b, 'full')
		axarr[i][j].plot(xpos,c,color='k')
		axarr[i][j].set_xlim(xpos[0],xpos[-1])
		if i == 0 : axarr[i][j].set_title(labels[j],fontsize=25)
		if j == 0 : axarr[i][j].set_ylabel(labels[i],fontsize=25)
		j += 1
	i += 1

fig.tight_layout()
fig.savefig('figs_tuning/tcurvexcorrs_'+str(totalstim)+'ms_allcells_Compare.png',dpi=300,transparent=True)
axarr[0][0].set_xlim(-100,100)
fig.savefig('figs_tuning/tcurvexcorrs_'+str(totalstim)+'ms_allcells_Compare_Zoomed.png',dpi=300,transparent=True)
