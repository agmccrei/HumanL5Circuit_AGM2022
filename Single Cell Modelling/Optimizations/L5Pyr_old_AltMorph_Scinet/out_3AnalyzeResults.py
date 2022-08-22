import numpy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pprint
import matplotlib._pylab_helpers
import matplotlib.patheffects as path_effects
import math
import scipy.stats as st
import statsmodels.stats.multitest as mt
import pandas as pd
import glob
import neuron
import seaborn as sns
from neuron import h, gui
import sys
import copy
import efel
import bluepyopt as bpop
import bluepyopt.ephys as ephys
pp = pprint.PrettyPrinter(indent=4)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)

pp.pprint("Initializing...")
sys.stdout.flush()

if sys.version_info[0] == 2:
	execfile("init_1morphology.py")
	execfile("init_2mechanisms.py")
	execfile("init_3parameters.py")
	execfile("init_4features.py")
	execfile("init_5recordings.py")
	execfile("init_6fitnesscalculator.py")
elif sys.version_info[0] == 3:
	exec(open("init_1morphology.py").read())
	exec(open("init_2mechanisms.py").read())
	exec(open("init_3parameters.py").read())
	exec(open("init_4features.py").read())
	exec(open("init_5recordings.py").read())
	exec(open("init_6fitnesscalculator.py").read())

vars = nonfrozen_params
objectives = list(all_features[0].keys()) + list(all_features[2].keys())
sagidx = objectives.index('sag_amplitude') # Example if also looking for error on a particular features
rmpidx = objectives.index('voltage_base') # Example if also looking for error on a particular features

# load results
finalpop_path = 'results/finalpop.pkl'
halloffame_path = 'results/halloffame.pkl'
hist_path = 'results/hist.pkl'
logs_path = 'results/logs.pkl'

finalpop = pickle.load(open(finalpop_path,"rb"))
halloffame = pickle.load(open(halloffame_path,"rb"))
hist = pickle.load(open(hist_path,"rb"))
logs = pickle.load(open(logs_path,"rb"))

# print params of chosen model
bestIndex = 0
for i2 in range(0,len(halloffame[bestIndex])):
	print(vars[i2]+': '+ str(halloffame[bestIndex][i2]))

# SD ranges
maxBaseSD = 2 # SD range
maxRMPSD = 2 # same as base SD
weights = numpy.ones(len(objectives))
for w in range(0,len(weights)):
	if objectives[w] == 'voltage_base': # Example if customizing
		weights[w] = weights[w]*maxRMPSD
	else:
		weights[w] = weights[w]*maxBaseSD

# iterate over all models across all generations to find highly ranked models that pass the SD threshold across all metrics
totalpop = []
qualityval = []
qualitysag = []
qualityrmp = []
for l in range(1,len(hist.__dict__['genealogy_history'])+1):
	f = hist.__dict__['genealogy_history'][l]
	count = 0
	for t in range(0,len(weights)):
		if f.fitness.values[t] < weights[t]:
			count += 1
	if count == len(weights):
		qualityval.append(numpy.sum(f.fitness.values))
		qualitysag.append(f.fitness.values[sagidx]) # Example if also looking for error on a particular features
		qualityrmp.append(f.fitness.values[rmpidx]) # Example if also looking for error on a particular features
		totalpop.append(f)
	else:
		continue

qualityval = numpy.array(qualityval)

fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(qualityval,50,facecolor='k', alpha=0.5,label='Young')
ax.set_xlabel('Sum of Standard Deviations')
ax.set_ylabel('Model Count')
fig.tight_layout()
fig.savefig('PLOTfiles/Quality.pdf', bbox_inches='tight')
fig.savefig('PLOTfiles/Quality.png', bbox_inches='tight')
plt.close(fig)

bestsag = abs(halloffame.__dict__['items'][bestIndex].__dict__['fitness'].__dict__['wvalues'][sagidx])
bestrmp = abs(halloffame.__dict__['items'][bestIndex].__dict__['fitness'].__dict__['wvalues'][rmpidx])

qualitysag = numpy.array(qualitysag)
qualityrmp = numpy.array(qualityrmp)

# Example variable errors
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(qualityrmp,qualitysag,color='k', alpha=0.7)
ax.scatter(bestrmp,bestsag,s=500,linewidths=2,edgecolors='k',color='darkgray', alpha=1)
for i2 in range(0,len(halloffame.__dict__['items'])):
	sag = abs(halloffame.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][sagidx])
	rmp = abs(halloffame.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][rmpidx])
	text1 = ax.text(rmp,sag,str(i2+1),color='white',horizontalalignment='center',verticalalignment='center',fontsize=20)
	text2 = ax.text(rmp,sag,str(i2+1),color='mistyrose',horizontalalignment='center',verticalalignment='center',fontsize=20)
	text1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
					   path_effects.Normal()])
	text2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='darkred'),
					   path_effects.Normal()])

bottom, top = ax.get_ylim()
ax.set_ylim((bottom,top))
ax.set_xlabel('RMP Error (SD)')
ax.set_ylabel('Sag Amplitude Error (SD)')
fig.tight_layout()
fig.savefig('PLOTfiles/Quality_SagVsRMP.pdf', dpi=300, bbox_inches='tight',transparent=True)
fig.savefig('PLOTfiles/Quality_SagVsRMP.png', dpi=300, bbox_inches='tight',transparent=True)
plt.close(fig)

# Variable distributions in highly-ranked models
for i in range(0,len(totalpop[0])):
	fp = numpy.transpose(totalpop)[i]
	plt.hist(fp,50,facecolor='k', alpha=0.5)
	bottom, top = plt.ylim()
	plt.plot([halloffame[bestIndex][i],halloffame[bestIndex][i]],[bottom,top],linestyle='dashed',color='k')
	plt.ylim((bottom,top))
	plt.xlabel(vars[i])
	plt.tight_layout()
	plt.savefig('PLOTfiles/' + vars[i] + '.pdf', bbox_inches='tight')
	plt.savefig('PLOTfiles/' + vars[i] + '.png', bbox_inches='tight')
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()

# Plot quality metrics over generations
gen_numbers = logs.select('gen')
min_fitness = numpy.array(logs.select('min'))
max_fitness = logs.select('max')
mean_fitness = numpy.array(logs.select('avg'))
std_fitness = numpy.array(logs.select('std'))

fig, ax = plt.subplots(1, figsize=(8, 8), facecolor='white')

stdminus = mean_fitness - std_fitness
stdplus = mean_fitness + std_fitness

ax.plot(
	gen_numbers,
	mean_fitness,
	color='black',
	linewidth=2,
	label='Population Average')

ax.fill_between(
	gen_numbers,
	min_fitness,
	max_fitness,
	color='black',
	alpha=0.4,
	linewidth=2,
	label=r'Population Standard Deviation')

ax.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
ax.set_xlabel('Generation #')
ax.set_ylabel('# Standard Deviations')
# ax.set_ylim([10, 100000])
ax.set_yscale('log')
plt.savefig('PLOTfiles/Performance.pdf', bbox_inches='tight', dpi=300)
plt.savefig('PLOTfiles/Performance.png', bbox_inches='tight', dpi=300)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
