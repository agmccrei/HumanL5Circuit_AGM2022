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
		'size'   : 28}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

controlm = ['y','o']
controli = ['basalX0.7','apic','basal','apicX1.5']
controli_labels = ['Low\nBasal','Low\nApical','Basal','Apical']
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

synnums = [85]
N_seeds = 30
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

df = pd.DataFrame(columns=["Cells",
			"Stim Loc",
			"Metric",
			"Mean Young",
			"SD Young",
			"Mean Old",
			"SD Old",
			"t-stat",
			"p-value",
			"Cohen's d"])

df2 = pd.DataFrame(columns=["Cells",
			"Stim Loc",
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

snr_m = []
snr_sd = []
snr_pval = []
snr_cd = []

rr_m = []
rr_sd = []
rr_pval = []
rr_cd = []

for controi in controli:
	Ratesy_allseeds1 = [[] for _ in range(0,len(rndinds))]
	Rateso_allseeds1 = [[] for _ in range(0,len(rndinds))]
	SNRy_allseeds1 = [[] for _ in range(0,len(rndinds))]
	SNRo_allseeds1 = [[] for _ in range(0,len(rndinds))]
	BaseRatey_allseeds1 = [[] for _ in range(0,len(rndinds))]
	BaseRateo_allseeds1 = [[] for _ in range(0,len(rndinds))]
	
	for control in controlm:
		if control == 'y': colc = 'k'
		if control == 'o': colc = 'r'
		path1 = control + '_' + str(synnums[0]) + '_' + controi + '/HL5netLFPy/Circuit_output/'
		
		for idx in range(0,len(rndinds)):
			print('Seed #'+str(idx))
			temp_sn1 = np.load(path1 + 'SPIKES_CircuitSeed1234StimSeed'+str(rndinds[idx])+'.npy',allow_pickle=True)
			
			# Only analyze PNs here
			SPIKES1 = [x for _,x in sorted(zip(temp_sn1.item()['gids'][0],temp_sn1.item()['times'][0]))]
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
		rvy = np.mean(BaseRatey_allseeds1[i])
		rvo = np.mean(BaseRateo_allseeds1[i])
		MeanBaseRatey.append(rvy)
		MeanBaseRateo.append(rvo)
		rvy = np.mean(SNRy_allseeds1[i])
		rvo = np.mean(SNRo_allseeds1[i])
		SNRy.append(rvy)
		SNRo.append(rvo)
		rvy = np.array(Ratesy_allseeds1[i])
		rvo = np.array(Rateso_allseeds1[i])
		PercentActivey.append((len(rvy[rvy>0])/N_HL5PN)*100)
		PercentActiveo.append((len(rvo[rvo>0])/N_HL5PN)*100)
		rvy = np.std(Ratesy_allseeds1[i])/np.mean(Ratesy_allseeds1[i])
		rvo = np.std(Rateso_allseeds1[i])/np.mean(Rateso_allseeds1[i])
		RateCVy.append(rvy)
		RateCVo.append(rvo)


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

	x = bs.bootstrap(np.array(MeanRatey1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	mryu1 = x.value
	x = bs.bootstrap(np.array(MeanRateo1), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	mrou1 = x.value

	# Plot tuning curves
	CI_means_PNy1 = []
	CI_lower_PNy1 = []
	CI_upper_PNy1 = []
	CI_means_PNo1 = []
	CI_lower_PNo1 = []
	CI_upper_PNo1 = []

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
	else:
		for l in range(0,N_HL5PN):
			CI_means_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l]))
			CI_lower_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l])-np.std(np.transpose(Ratesy_allseeds1)[l]))
			CI_upper_PNy1.append(np.mean(np.transpose(Ratesy_allseeds1)[l])+np.std(np.transpose(Ratesy_allseeds1)[l]))
			CI_means_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l]))
			CI_lower_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l])-np.std(np.transpose(Rateso_allseeds1)[l]))
			CI_upper_PNo1.append(np.mean(np.transpose(Rateso_allseeds1)[l])+np.std(np.transpose(Rateso_allseeds1)[l]))
	
	tstat_S0, pval_S0 = st.ttest_rel(MeanBaseRatey,MeanRatey)
	tstat_S1, pval_S1 = st.ttest_rel(MeanBaseRateo,MeanRateo)
	tstat_S2, pval_S2 = st.ttest_rel(MeanBaseRatey,MeanBaseRateo)
	tstat_S3, pval_S3 = st.ttest_rel(MeanRatey,MeanRateo)
	
	x = [0.1,0.9,2.1,2.9]
	x2 = [0.5,2.5]
	fig = plt.figure(figsize=(7,5))
	ax1 = fig.add_subplot(111)
	ax1.bar(x, [meanbaseratey,meanbaserateo,meanratey,meanrateo], yerr=[[lay0,lao0],[uay0,uao0],[lay,lao],[uay,uao]] if bsMeans else [lowerbaseratey,lowerbaserateo,lowerratey,lowerrateo], color=['darkgray','lightcoral','darkgray','lightcoral'], width=0.8,tick_label=['Young','Old','Young','Old'],linewidth=4,ecolor='black',error_kw={'elinewidth':4,'markeredgewidth':4},capsize=12,edgecolor='k')
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
	ax1.set_xticks(x2)
	ax1.set_xticklabels(['Pre','Post'])
	ax1.set_xlim(-0.6,3.6)
	ax1.set_ylim(0,13)
	ax1.spines["right"].set_visible(False)
	ax1.spines["top"].set_visible(False)
	fig.tight_layout()
	fig.savefig('figs_tuning/MeanBaseVsResponseRates_'+controi+'_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
	plt.close(fig)
	
	gscalery1 = np.percentile(CI_means_PNy1,90)-mryu1
	gscalero1 = np.percentile(CI_means_PNo1,90)-mrou1
	mu_ind1 = int(N_HL5PNso*synnums[0]/180) # find index with peak input depending on orientation
	sigma = int(N_HL5PNso*42/180) # Set sigma to 42 degrees (Bensmaia et al 2008) and scale by number of PN neurons

	x_values = np.linspace(0, N_HL5PN-1, N_HL5PN)
	gfuncy1 = gaussian(x_values, mu_ind1, sigma)*gscalery1+mryu1 # Gaussian vector (y=[0,1]) of length = cellnum
	gfuncy1[x_values>N_HL5PNso]=mryu1
	gfunco1 = gaussian(x_values, mu_ind1, sigma)*gscalero1+mrou1 # Gaussian vector (y=[0,1]) of length = cellnum
	gfunco1[x_values>N_HL5PNso]=mrou1

	xcells = np.arange(0,N_HL5PN,1)
	ymax = np.max([np.max(CI_upper_PNy1),np.max(CI_upper_PNo1)])

	fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(10,5),sharex=True,sharey=True)
	axarr.scatter(xcells,CI_means_PNy1,c='k',alpha=0.25,label=str(synnums[0])+r'$\degree$', linewidths=1, edgecolors='none')
	axarr.scatter(xcells,CI_means_PNo1,c='lightcoral',alpha=0.25,label=str(synnums[0])+r'$\degree$', linewidths=1, edgecolors='none')
	axarr.plot(x_values,gfuncy1,color='k',alpha=1,lw=3,ls=':')
	axarr.plot(x_values,gfunco1,color='lightcoral',alpha=1,lw=3,ls=':')
	axarr.set_xlim(0,N_HL5PN)
	axarr.set_ylim(0,ymax)
	ylims=axarr.get_ylim()
	axarr.plot(np.array([N_HL5PNso,N_HL5PNso]),ylims,color='dimgray',alpha=1,lw=2,ls='dashed')
	axarr.set_ylim(0,ymax)
	axarr.set_ylabel('Pyr Response Rate (Hz)')
	axarr.set_xlim(0,N_HL5PN)
	axarr.set_xlabel('Neuron Index')
	fig.tight_layout()
	fig.savefig('figs_tuning/tuningcurve_scatter_YoungOnlyA1_'+controi+'_'+str(totalstim)+'ms_allcells_Compare.png',dpi=300,transparent=True)
	
	lay = abs(meanSNRy-lowerSNRy)
	uay = abs(meanSNRy-upperSNRy)
	lao = abs(meanSNRo-lowerSNRo)
	uao = abs(meanSNRo-upperSNRo)
	
	tstat_S, pval_S = st.ttest_rel(SNRy,SNRo)
	cd = cohen_d(SNRy,SNRo)
	
	snr_m.append([meanSNRy,meanSNRo])
	snr_sd.append([[lay,lao],[uay,uao]] if bsMeans else [lowerSNRy,lowerSNRo])
	snr_pval.append(pval_S)
	snr_cd.append(cd)
	
	df = df.append({"Cells" : 'All',
				"Stim Loc" : controi,
				"Metric" : 'SNR',
				"Mean Young" : meanSNRy,
				"SD Young" : [lay,lay] if bsMeans else lowerSNRy,
				"Mean Old" : meanSNRo,
				"SD Old" : [lao,lao] if bsMeans else lowerSNRo,
				"t-stat" : tstat_S,
				"p-value" : pval_S,
				"Cohen's d" : cd},
				ignore_index = True)
	
	lay0 = abs(meanbaseratey-lowerbaseratey)
	uay0 = abs(meanbaseratey-upperbaseratey)
	lao0 = abs(meanbaserateo-lowerbaserateo)
	uao0 = abs(meanbaserateo-upperbaserateo)
	
	lay = abs(meanratey-lowerratey)
	uay = abs(meanratey-upperratey)
	lao = abs(meanrateo-lowerrateo)
	uao = abs(meanrateo-upperrateo)
		
	cd0 = cohen_d(MeanBaseRatey,MeanRatey)
	cd1 = cohen_d(MeanBaseRateo,MeanRateo)
	cd2 = cohen_d(MeanBaseRatey,MeanBaseRateo)
	cd3 = cohen_d(MeanRatey,MeanRateo)
	
	df2 = df2.append({"Cells" : 'All',
				"Stim Loc" : controi,
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
	
	rr_m.append([meanratey,meanrateo])
	rr_sd.append([[lay,lao],[uay,uao]] if bsMeans else [lowerratey,lowerrateo])
	rr_pval.append(pval_S3)
	rr_cd.append(cd3)

# Save stats dataframe to csv
df.to_csv('figs_tuning/stats_'+str(totalstim)+'ms.csv')
df2.to_csv('figs_tuning/stats_BaseVsResponseRates_'+str(totalstim)+'ms.csv')

x = [[0.1+i*2,0.9+i*2] for i in range(len(controli))]
x2 = [np.sum(x_)/2 for x_ in x]
fig = plt.figure(figsize=(8,5)) # (14,6)
ax1 = fig.add_subplot(111)
for ind,_ in enumerate(controli):
	ax1.bar(x[ind], snr_m[ind], yerr=snr_sd[ind], color=['darkgray','lightcoral'], width=0.8,tick_label=['Young','Old'],linewidth=1,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
	print('SNR p-value='+str(snr_pval[ind]))
	if snr_pval[ind] < 0.001:
		barplot_annotate_brackets(0, 1, '***', [x[ind][i] for i in [0, 1]], snr_m[ind], yerr=snr_sd[ind])
	elif snr_pval[ind] < 0.01:
		barplot_annotate_brackets(0, 1, '**', [x[ind][i] for i in [0, 1]], snr_m[ind], yerr=snr_sd[ind])
	elif snr_pval[ind] < 0.05:
		barplot_annotate_brackets(0, 1, '*', [x[ind][i] for i in [0, 1]], snr_m[ind], yerr=snr_sd[ind])
ax1.set_xticks(x2)
ax1.set_xticklabels(controli_labels)
ax1.set_ylabel('Pyr SNR')
ax1.set_xlim( x[0][0]-0.7 , x[-1][-1]+0.7 )
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/SNR_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

x = [[0.1+i*2,0.9+i*2] for i in range(len(controli))]
x2 = [np.sum(x_)/2 for x_ in x]
fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
for ind,_ in enumerate(controli):
	ax1.bar(x[ind], rr_m[ind], yerr=rr_sd[ind], color=['darkgray','lightcoral'], width=0.8,tick_label=['Young','Old'],linewidth=1,ecolor='black',error_kw={'elinewidth':3,'markeredgewidth':3},capsize=12,edgecolor='k')
	print('Response Rate p-value='+str(snr_pval[ind]))
	if rr_pval[ind] < 0.001:
		barplot_annotate_brackets(0, 1, '***', [x[ind][i] for i in [0, 1]], rr_m[ind], yerr=rr_sd[ind])
	elif rr_pval[ind] < 0.01:
		barplot_annotate_brackets(0, 1, '**', [x[ind][i] for i in [0, 1]], rr_m[ind], yerr=rr_sd[ind])
	elif rr_pval[ind] < 0.05:
		barplot_annotate_brackets(0, 1, '*', [x[ind][i] for i in [0, 1]], rr_m[ind], yerr=rr_sd[ind])
ax1.set_xticks(x2)
ax1.set_xticklabels(controli_labels)
ax1.set_ylabel('Pyr Response Rate')
ax1.set_xlim( x[0][0]-0.7 , x[-1][-1]+0.7 )
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/ResponseRates_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(111)
ax1.bar(x2, snr_cd, color='darkgray', width=1.5,tick_label=controli_labels,linewidth=4,edgecolor='k')
ax1.set_ylabel("Cohen's D")
ax1.set_xlim( x[0][0]-0.7 , x[-1][-1]+0.7 )
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig('figs_tuning/Conhensd_'+str(totalstim)+'ms_AllCellsDuringStim_MergedOrientations.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)
