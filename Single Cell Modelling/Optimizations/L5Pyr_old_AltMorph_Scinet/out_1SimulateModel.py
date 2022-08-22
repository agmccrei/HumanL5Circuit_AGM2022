import time
import numpy
import matplotlib.pyplot as plt
from neuron import h, gui
import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pprint
import copy
import efel
import os
import sys
import math
from math import log10, floor
import collections
import pickle
pp = pprint.PrettyPrinter(indent=4)
TotalStartTime = time.time()

pp.pprint("Initializing...")
sys.stdout.flush()

idx = 0 # which model to select from halloffame (0 = best)

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

#################
### Run Model ###
#################

finalpop_path = 'results/finalpop.pkl'
halloffame_path = 'results/halloffame.pkl'
hist_path = 'results/hist.pkl'
logs_path = 'results/logs.pkl'

finalpop = pickle.load(open(finalpop_path,"rb"))
halloffame = pickle.load(open(halloffame_path,"rb"))
hist = pickle.load(open(hist_path,"rb"))
logs = pickle.load(open(logs_path,"rb"))

##### Select model from hall of fame where 0 is the most highly ranked #####
best_ind = halloffame[idx]

##### Print optimized params to text file #####
text_file = open('work/N' + str(idx+1) + 'params.txt', "w")
text_file.write('\nBest #' + str(idx+1) + ': \n')
text_file.write(pp.pformat(list(zip(nonfrozen_params,best_ind))))
text_file.write('\nBest #' + str(idx+1) + ' Fitness values: \n')
text_file.write(str(sum(best_ind.fitness.values)))

##### Print model hoc code to file #####
best_ind_dict = cell_evaluator.param_dict(best_ind)

hc = Cell_Model.create_hoc(param_values=best_ind_dict)
text_file = open('work/template' + str(idx) + '.hoc', "w")
text_file.write(hc)

##### Run model #####
responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict, sim=nrn)

# Align experimental voltage traces to the resting of the first trace
if single_cell_data == True:
	for i in range(1,len(data)):
		ind1 = numpy.where(data[0]['T'] == 270)
		ind2 = numpy.where(data[i]['T'] == 270)
		shift1 = 0
		shift2 = data[i]['V'][ind2]-data[0]['V'][ind1]
		data[i]['V'] = data[i]['V']-shift2

##### Plot Model #####
NumSomaRecs = numpy.sum([i=='soma' for i in RecLocs])
def plot_responses(responses):
	fig, axarr = plt.subplots(nrows=NumSomaRecs,ncols=1, sharex=True)
	for i in range(0,NumSomaRecs):
		if single_cell_data == True:
			axarr[i].plot(data[i]['T'],data[i]['V'], color='b')
		axarr[i].plot(responses['step'+str(StepNums[i])+'.soma.v']['time'], responses['step'+str(StepNums[i])+'.soma.v']['voltage'], color='r')
		axarr[i].set_xlim(stimstart-200,stimend+200)
	fig.tight_layout()

def plot_responses_axon(responses):
	fig, axarr = plt.subplots(nrows=NumSomaRecs,ncols=1, sharex=True)
	for i in range(0,NumSomaRecs):
		axarr[i].plot(responses['step'+str(StepNums[i])+'.axon.v']['time'], responses['step'+str(StepNums[i])+'.axon.v']['voltage'], color='m')
		axarr[i].plot(responses['step'+str(StepNums[i])+'.soma.v']['time'], responses['step'+str(StepNums[i])+'.soma.v']['voltage'], color='r')
		axarr[i].set_xlim(stimstart,stimstart+200)
	fig.tight_layout()

plot_responses(responses)
plt.savefig('PLOTfiles/Post_Traces' + str(idx+1) + '.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('PLOTfiles/Post_Traces' + str(idx+1) + '.png', bbox_inches='tight', dpi=300, transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/Post_Traces' + str(idx+1) + '_axon.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('PLOTfiles/Post_Traces' + str(idx+1) + '_axon.png', bbox_inches='tight', dpi=300, transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
