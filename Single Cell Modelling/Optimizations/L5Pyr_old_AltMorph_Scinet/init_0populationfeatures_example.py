# Uses data from https://github.com/stripathy/valiante_ih/tree/master/summary_tables
# Loads 'cell_patient_ephys_combined.csv' and then downloads and analyzes experimental traces from the repository using entries from that file.
import urllib.request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import math
from ipfx.sweep import Sweep, SweepSet
import os.path
from os import path
import efel
import pprint
pp = pprint.PrettyPrinter(indent=4)
efel.api.setThreshold(-20)

Features_act = ['Spikecount', 'mean_frequency', 'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_slow_time', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','AP_duration','AP_fall_time','steady_state_voltage_stimend','decay_time_constant_after_stim']
Features_pas = ['sag_amplitude']
JPC = 10.8 # For junction potential correction
LayerOfInterest = 'L5'
CellOfInterest = 'Pyr'
Recorder = 'Homeira'
CurrentStep = 300 # pA - Specify which current step to extract summary info from
if CurrentStep < 0:
	Features = Features_pas
else:
	Features = Features_act

OldGroup = '>= 50'
YngGroup = '< 50'
AllGroup = '> 0'
AgeOperator = AllGroup

def myround(x, base=50):
	return base * round(x/base)

# download file from github repo
filename = 'cell_patient_ephys_combined.csv'
data = pd.read_csv (filename)
df1 = pd.DataFrame(data, columns= ['age'])
df2 = pd.DataFrame(data, columns= ['cell_id'])
df3 = pd.DataFrame(data, columns= ['layer_name'])
df4 = pd.DataFrame(data, columns= ['cell_type'])
df5 = pd.DataFrame(data, columns= ['recorder_name'])

agevec0 = df1.to_numpy()
cidvec0 = df2.to_numpy()
layvec0 = df3.to_numpy()
typvec0 = df4.to_numpy()
namvec0 = df5.to_numpy()

agevec = []
cidvec = []

features_values_allcells = []
print(str(len(cidvec0)),' traces')
for i in range(0,len(cidvec0)):
	if ((eval('agevec0[i][0] ' + AgeOperator)) & (layvec0[i][0] == LayerOfInterest) & (typvec0[i][0] == CellOfInterest) & (namvec0[i][0] == Recorder)):
		agevec.append(agevec0[i][0])
		cidvec.append(cidvec0[i][0])
		
		intrinsic_file_name = cidvec[len(cidvec)-1]
		url = "https://github.com/stripathy/valiante_lab_abf_process/blob/master/sweep_sets/%s.p?raw=true" % (intrinsic_file_name) # dl=1 is important
		
		filename = intrinsic_file_name + '.p'
		
		# If file already downloaded then just load it; else load from url
		if path.exists(filename):
			# load the file into a python object called sweep_set
			with open(filename, 'rb') as fp:
				sweep_set = pickle.load(fp)
		else:
			try:
				u = urllib.request.urlopen(url)
				data = u.read()
				u.close()
				with open(filename, "wb") as f :
					f.write(data)
				
				# load the file into a python object called sweep_set
				with open(filename, 'rb') as fp:
					sweep_set = pickle.load(fp)
				
			except:
				print(filename + ' does not exist on the Github')
				continue
		
		for c in range(0,len(sweep_set.sweeps)):
			minc = myround(np.amin(sweep_set.sweeps[c].i))
			maxc = myround(np.amax(sweep_set.sweeps[c].i))
			if abs(minc) > abs(maxc):
				curamp = minc
			elif abs(minc) < abs(maxc):
				curamp = maxc
			else:
				curamp = 0
			
			if (curamp<CurrentStep-25):
				continue
			if (curamp>CurrentStep+25):
				continue
			
			step = sweep_set.sweeps[c]
			# Find stim start
			for k in range(0,len(step.i)):
				if ((step.i[k] < -10) or (step.i[k] > 10)):
					tstart = step.t[k-1]
					break
			
			# Find stim end
			for k in range(len(step.i)-1,0,-1):
				if ((step.i[k] < -10) or (step.i[k] > 10)):
					tstop = step.t[k+1]
					break
			
			trace = {}
			trace['T'] = step.t*1000
			trace['V'] = step.v-JPC
			trace['stim_start'] = [tstart*1000]
			trace['stim_end'] = [tstop*1000]
			trace['name'] = [CurrentStep]
			trace['stimulus_current'] = [curamp/1000]
			features_values = efel.getMeanFeatureValues([trace], Features)
			features_values[0]['cell_id'] = cidvec0[i][0]
			features_values[0]['age'] = agevec0[i][0]
			features_values[0]['current_step'] = curamp
			
			# Only take steps of a particular duration
			stimduration = trace['stim_end'][0]-trace['stim_start'][0]
			if ((stimduration>601) or (stimduration<595)):
				continue
			
			if agevec0[i][0] < 50: plt.plot(step.t*1000,step.v-JPC,'k')
			if agevec0[i][0] >= 50: plt.plot(step.t*1000,step.v-JPC,'r')
			plt.xlim(tstart*1000 - 80,tstop*1000 + 250)
			plt.ylabel('Voltage (mV)')
			plt.xlabel('Time (ms)')
			plt.savefig('figs_voltages/' + LayerOfInterest + CellOfInterest + '_' + AgeOperator + '_' + 'cell' + str(i) + '_' + str(CurrentStep) + 'pA.png')
			plt.close()
			
			features_values_allcells.append(features_values[0])

for feat in Features:
	featuresvals = [d[feat] for d in features_values_allcells]
	featuresvals = [np.nan if v is None else v for v in featuresvals]
	featuremean = np.nanmean(featuresvals)
	featurestdv = np.nanstd(featuresvals)
	print(feat + ' mean = ' + str(featuremean) + ' +/- ' + str(featurestdv))
	
