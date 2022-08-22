#################################
### Print and Plot Top Models ###
#################################
best_ind = halloffame[0]
best_ind2 = halloffame[1]
best_ind3 = halloffame[2]
best_ind4 = halloffame[3]
best_ind5 = halloffame[4]

print('Best individual: ', list(zip(nonfrozen_params,best_ind)))
sys.stdout.flush()
print('2nd individual: ', list(zip(nonfrozen_params,best_ind2)))
sys.stdout.flush()
print('3rd individual: ', list(zip(nonfrozen_params,best_ind3)))
sys.stdout.flush()
print('4th individual: ', list(zip(nonfrozen_params,best_ind4)))
sys.stdout.flush()
print('5th individual: ', list(zip(nonfrozen_params,best_ind5)))
sys.stdout.flush()

print('Best Fitness values: ', sum(best_ind.fitness.values))
sys.stdout.flush()
print('2nd Fitness values: ', sum(best_ind2.fitness.values))
sys.stdout.flush()
print('3rd Fitness values: ', sum(best_ind3.fitness.values))
sys.stdout.flush()
print('4th Fitness values: ', sum(best_ind4.fitness.values))
sys.stdout.flush()
print('5th Fitness values: ', sum(best_ind5.fitness.values))
sys.stdout.flush()

best_ind_dict = cell_evaluator.param_dict(best_ind)
best_ind_dict2 = cell_evaluator.param_dict(best_ind2)
best_ind_dict3 = cell_evaluator.param_dict(best_ind3)
best_ind_dict4 = cell_evaluator.param_dict(best_ind4)
best_ind_dict5 = cell_evaluator.param_dict(best_ind5)

# Align experimental voltage traces to the resting of the first trace
if target_feature_type == 'Automatic':
	for i in range(1,len(data)):
		ind1 = numpy.where(data[0]['T'] == 270)
		ind2 = numpy.where(data[i]['T'] == 270)
		shift1 = 0
		shift2 = data[i]['V'][ind2]-data[0]['V'][ind1]
		data[i]['V'] = data[i]['V']-shift2

##### Plot Top Models #####
NumSomaRecs = numpy.sum([i=='soma' for i in RecLocs])
def plot_responses(responses):
	fig, axarr = plt.subplots(nrows=NumSomaRecs,ncols=1, sharex=True)
	for i in range(0,NumSomaRecs):
		if target_feature_type == 'Automatic':
			axarr[i].plot(data[i]['T'],data[i]['V'], color='b')
		axarr[i].plot(responses[stepnames[i]+'.soma.v']['time'], responses[stepnames[i]+'.soma.v']['voltage'], color='r')
		axarr[i].set_xlim(stimstart-200,stimend+200)
	fig.tight_layout()

def plot_responses_axon(responses):
	fig, axarr = plt.subplots(nrows=NumSomaRecs,ncols=1, sharex=True)
	for i in range(0,NumSomaRecs):
		axarr[i].plot(responses[stepnames[i]+'.axon.v']['time'], responses[stepnames[i]+'.axon.v']['voltage'], color='m')
		axarr[i].plot(responses[stepnames[i]+'.soma.v']['time'], responses[stepnames[i]+'.soma.v']['voltage'], color='r')
		axarr[i].set_xlim(stimstart,stimstart+200)
	fig.tight_layout()

# Feature calculator function when inputting model traces
def get_features2(responses):
	# Only take simdata used in fitness evaluator (i.e. StepNums variable)
	simdata = collections.OrderedDict()
	for l,i in enumerate(StepNums):
		recname = 'step'+str(i)+'.'+RecLocs[i]+'.v'
		sweep_data = {}
		sweep_data['V'] = responses[recname]['voltage']
		sweep_data['T'] = responses[recname]['time']
		simdata[l] = sweep_data
	
	features_values_mean = []
	for step_name, step_traces in simdata.items():
		trace = {}
		trace['T'] = simdata[step_name]['T']
		trace['V'] = simdata[step_name]['V']
		trace['stim_start'] = [stimstart]
		trace['stim_end'] = [stimend]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		feature_values = efel.getMeanFeatureValues([trace], AllFeatNames[step_name])
		features_values_mean.extend(feature_values)
	
	return list(features_values_mean)

# Error calculator function when inputting model traces
def get_stdevs(simfeats):
	num_stdevs = []
	for t in range(len(all_features)):
		err={}
		for feat in all_features[t]:
			if simfeats[t][feat] is not None:
				err[feat] = abs((simfeats[t][feat]-all_features[t][feat])/all_features_sd[t][feat])
			else:
				err[feat] = None
		
		num_stdevs.append(err)
	
	return num_stdevs

responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict, sim=nrn)
plot_responses(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_axon.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_axon.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

all_features1 = get_features2(responses)

print("\nTop Model 1 Features:\n")
sys.stdout.flush()
pp.pprint(all_features1)
sys.stdout.flush()

active_errs1 = get_stdevs(all_features1)

print("\nTop Model 1 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs1)
sys.stdout.flush()

responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict2, sim=nrn)
plot_responses(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_axon.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_axon.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

all_features2 = get_features2(responses)

print("\nTop Model 2 Features:\n")
sys.stdout.flush()
pp.pprint(all_features2)
sys.stdout.flush()

active_errs2 = get_stdevs(all_features2)

print("\nTop Model 2 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs2)
sys.stdout.flush()

responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict3, sim=nrn)
plot_responses(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_axon.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_axon.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

all_features3 = get_features2(responses)

print("\nTop Model 3 Features:\n")
sys.stdout.flush()
pp.pprint(all_features3)
sys.stdout.flush()

active_errs3 = get_stdevs(all_features3)

print("\nTop Model 3 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs3)
sys.stdout.flush()

responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict4, sim=nrn)
plot_responses(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_axon.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_axon.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

all_features4 = get_features2(responses)

print("\nTop Model 4 Features:\n")
sys.stdout.flush()
pp.pprint(all_features4)
sys.stdout.flush()

active_errs4 = get_stdevs(all_features4)

print("\nTop Model 4 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs4)
sys.stdout.flush()

responses = recording_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict5, sim=nrn)
plot_responses(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_axon(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_axon.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_axon.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

all_features5 = get_features2(responses)

print("\nTop Model 5 Features:\n")
sys.stdout.flush()
pp.pprint(all_features5)
sys.stdout.flush()

active_errs5 = get_stdevs(all_features5)

print("\nTop Model 5 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs5)
sys.stdout.flush()

print("\nExperimental Features:\n")
sys.stdout.flush()
pp.pprint(all_features)
sys.stdout.flush()

##### Plot Performance ######
gen_numbers = logs.select('gen')
min_fitness = numpy.array(logs.select('min'))
max_fitness = logs.select('max')
mean_fitness = numpy.array(logs.select('avg'))
std_fitness = numpy.array(logs.select('std'))

fig, ax = plt.subplots(1, figsize=(8, 8), facecolor='white')

std = std_fitness
mean = mean_fitness
minimum = min_fitness
stdminus = mean - std
stdplus = mean + std

ax.plot(
	gen_numbers,
	mean,
	color='black',
	linewidth=2,
	label='Population Average')

ax.fill_between(
	gen_numbers,
	stdminus,
	stdplus,
	color='lightgray',
	linewidth=2,
	label=r'Population Standard Deviation')

ax.plot(
	gen_numbers,
	minimum,
	color='red',
	linewidth=2,
	label='Population Minimum')

ax.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
ax.set_xlabel('Generation #',fontsize = 10)
ax.set_ylabel('# Standard Deviations',fontsize = 10)
ax.set_ylim([0, max(stdplus)])
ax.legend()
plt.savefig('PLOTfiles/' + cellname + '_Performance.pdf', bbox_inches='tight',dpi=300,transparent=True)
plt.savefig('PLOTfiles/' + cellname + '_Performance.png', bbox_inches='tight',dpi=300,transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
