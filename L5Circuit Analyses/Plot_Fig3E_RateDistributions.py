################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
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
import pandas as pd

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

manipulation = 'Old'
condition = 'dend'
normal = '1_y_'+condition+'_0'
condit = '1_o_'+condition+'_0'
normalpath = normal + '/HL5netLFPy/Circuit_output/'
conditpath = condit + '/HL5netLFPy/Circuit_output/'

colors = ['dimgrey', 'red', 'green', 'orange']

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

instarates1_PN = np.array([])
instarates2_PN = np.array([])

instarates1_PN_perseed = []
instarates2_PN_perseed = []

bin_size_insta = 0.25
max_insta = 20

for i in N_seedsList:
	print('Testing Seed #'+str(i))
	
	temp_sn = np.load(normalpath + 'SPIKES_Seed' + str(i) + '.npy',allow_pickle=True)
	temp_cn = np.load(conditpath + 'SPIKES_Seed' + str(i) + '.npy',allow_pickle=True)
	
	# only pyramidal - condition 1
	SPIKES = temp_sn.item()
	ir_temp = np.array([])
	for j in range(0,len(SPIKES['times'][0])):
		scount = SPIKES['times'][0][j][(SPIKES['times'][0][j]>startsclice) & (SPIKES['times'][0][j]<=stimtime)]
		diff1 = 1000/np.diff(scount) # also converts ms to s
		instarates1_PN = np.concatenate((instarates1_PN,diff1))
		ir_temp = np.concatenate((ir_temp,diff1))
	
	ISI_dist, bins1 = np.histogram(ir_temp,bins=int(max_insta/bin_size_insta),range=(0,max_insta),density=False)
	instarates1_PN_perseed.append(ISI_dist)
	
	# only pyramidal - condition 2
	SPIKES = temp_cn.item()
	ir_temp = np.array([])
	for j in range(0,len(SPIKES['times'][0])):
		scount = SPIKES['times'][0][j][(SPIKES['times'][0][j]>startsclice) & (SPIKES['times'][0][j]<=stimtime)]
		diff1 = 1000/np.diff(scount) # also converts ms to s
		instarates2_PN = np.concatenate((instarates2_PN,diff1))
		ir_temp = np.concatenate((ir_temp,diff1))
	
	numbins = int((np.max(ir_temp) - np.min(ir_temp))/bin_size_insta)
	ISI_dist, bins2 = np.histogram(ir_temp,bins=int(max_insta/bin_size_insta),range=(0,max_insta),density=False)
	instarates2_PN_perseed.append(ISI_dist)

nstat_PN1, npval_PN1 = st.normaltest(instarates1_PN)
nstat_PN2, npval_PN2 = st.normaltest(instarates2_PN)
mwu_stat, mwu_pval = st.mannwhitneyu(instarates1_PN,instarates2_PN)

df = pd.DataFrame(columns=["Comparison","Normal Test Young","Normal Test Old","MWU Stat","MWU P-Value"])
df = df.append({"Comparison" : 'Pooled ISIs of Pyr Neurons Across Seeds',
			"Normal Test Young" : npval_PN1,
			"Normal Test Old" : npval_PN2,
			"MWU Stat" : mwu_stat,
			"MWU P-Value" : mwu_pval},
			ignore_index = True)

numbins1 = int((np.max(instarates1_PN) - np.min(instarates1_PN))/bin_size_insta)
numbins2 = int((np.max(instarates2_PN) - np.min(instarates2_PN))/bin_size_insta)
maxdata = np.max(np.array([np.max(instarates1_PN),np.max(instarates2_PN)]))
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(111)
ax1.hist(instarates1_PN,bins=numbins1,color='white',alpha=0.8,edgecolor='k',label='Younger')
ax1.hist(instarates2_PN,bins=numbins2,color='white',alpha=0.8,edgecolor='tab:red',label='Older')
ax1.tick_params(axis='x', which='major', bottom=True)
ax1.tick_params(axis='y', which='major', left=True)
ax1.tick_params(axis='x', which='minor', bottom=True)
ax1.tick_params(axis='y', which='minor', left=True)
# ax1.xaxis.set_minor_locator(MultipleLocator(5))
# ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0,max_insta)
# ax1.set_yscale('log')
ax1.set_ylabel('# of ISIs')
ax1.set_xlabel('Rate (Hz)')
ax1.legend(loc='upper right')
fig.savefig('figs_RateDistributions/RateDistribution_PN.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

ISI1_dist, bins1 = np.histogram(instarates1_PN,bins=int(max_insta/bin_size_insta),range=(0,max_insta),density=False)
ISI2_dist, bins2 = np.histogram(instarates2_PN,bins=int(max_insta/bin_size_insta),range=(0,max_insta),density=False)
binsvec = np.linspace(0,max_insta, num=int(max_insta/bin_size_insta))
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(111)
ax1.plot(binsvec,ISI1_dist,linewidth=1.5,color='k',alpha=1,label='Younger')
ax1.plot(binsvec,ISI2_dist,linewidth=1.5,color='tab:red',alpha=1,label='Older')
ax1.tick_params(axis='x', which='major', bottom=True)
ax1.tick_params(axis='y', which='major', left=True)
ax1.tick_params(axis='x', which='minor', bottom=True)
ax1.tick_params(axis='y', which='minor', left=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0,max_insta)
ax1.set_ylim(bottom=0)
ax1.set_ylabel('# of ISIs')
ax1.set_xlabel('Rate (Hz)')
ax1.legend(loc='upper right')
fig.savefig('figs_RateDistributions/RateDistribution_line_PN.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

ir1_m = np.mean(instarates1_PN_perseed,axis=0)
ir2_m = np.mean(instarates2_PN_perseed,axis=0)
ir1_sd = np.std(instarates1_PN_perseed,axis=0)
ir2_sd = np.std(instarates2_PN_perseed,axis=0)
# Compute 95% confidence interval of spike PSD for PNs
CI_means_PN = []
CI_lower_PN = []
CI_upper_PN = []
CI_means_PNc = []
CI_lower_PNc = []
CI_upper_PNc = []
print('Starting Bootstrapping')
for l in range(0,len(instarates1_PN_perseed[0])):
	x = bs.bootstrap(np.transpose(instarates1_PN_perseed)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=100)
	CI_means_PN.append(x.value)
	CI_lower_PN.append(x.lower_bound)
	CI_upper_PN.append(x.upper_bound)
	x = bs.bootstrap(np.transpose(instarates2_PN_perseed)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=100)
	CI_means_PNc.append(x.value)
	CI_lower_PNc.append(x.lower_bound)
	CI_upper_PNc.append(x.upper_bound)
print('Bootstrapping Done')

nstat_PN1, npval_PN1 = st.normaltest(ir1_m)
nstat_PN2, npval_PN2 = st.normaltest(ir2_m)
mwu_stat, mwu_pval = st.mannwhitneyu(ir1_m,ir2_m)

df = df.append({"Comparison" : 'Mean Pyr Neuron ISI Distributions',
			"Normal Test Young" : npval_PN1,
			"Normal Test Old" : npval_PN2,
			"MWU Stat" : mwu_stat,
			"MWU P-Value" : mwu_pval},
			ignore_index = True)
df.to_csv('figs_RateDistributions/stats_MannWhitneyUTest_Rates.csv')

binsvec = np.linspace(0,max_insta, num=int(max_insta/bin_size_insta))

fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(111)
ax1.fill_between(binsvec, CI_lower_PN, CI_upper_PN,color='k',alpha=0.4)
ax1.fill_between(binsvec, CI_lower_PNc, CI_upper_PNc,color='r',alpha=0.4)
ax1.plot(binsvec, CI_means_PN, 'k',label='Younger')
ax1.plot(binsvec, CI_means_PNc, 'tab:red',label='Older')
ax1.tick_params(axis='x', which='major', bottom=True)
ax1.tick_params(axis='y', which='major', left=True)
ax1.tick_params(axis='x', which='minor', bottom=True)
ax1.tick_params(axis='y', which='minor', left=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0,max_insta)
ax1.set_ylim(bottom=0)
ax1.set_ylabel('# of ISIs')
ax1.set_xlabel('Rate (Hz)')
ax1.legend(loc='upper right')
fig.savefig('figs_RateDistributions/RateDistribution_PerSeed_bootstrappedCI_PN.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(111)
ax1.errorbar(binsvec,ir1_m,yerr=ir1_sd,capsize=1.5,linewidth=1.5,elinewidth=1.5,color='k',alpha=1,label='Younger')
ax1.errorbar(binsvec,ir2_m,yerr=ir2_sd,capsize=1.5,linewidth=1.5,elinewidth=1.5,color='tab:red',alpha=1,label='Older')
ax1.tick_params(axis='x', which='major', bottom=True)
ax1.tick_params(axis='y', which='major', left=True)
ax1.tick_params(axis='x', which='minor', bottom=True)
ax1.tick_params(axis='y', which='minor', left=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0,max_insta)
ax1.set_ylim(bottom=0)
ax1.set_ylabel('# of ISIs')
ax1.set_xlabel('Rate (Hz)')
ax1.legend(loc='upper right')
fig.savefig('figs_RateDistributions/RateDistribution_PerSeed_PN.png',bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)
