#################################
### Set-up Fitness Calculator ###
#################################
def define_fitness_calculator(sweeps,feature_definitions,feature_sds):
	"""Define fitness calculator"""
	
	features_all = []
	
	# Create feature list for somatic features
	for c, sweep in enumerate(sweeps):
		stim_start = sweep.stimuli[0].step_delay
		stim_end = stim_start + sweep.stimuli[0].step_duration
		stim_amp = sweep.stimuli[0].step_amplitude
		for efel_feature_name in feature_definitions[c]:
			feature_name = '%s.%s' % (sweep.name, efel_feature_name)
			mean = feature_definitions[c][efel_feature_name]
			sd = feature_sds[c][efel_feature_name]
			
			recording_names = {'': '%s.soma.v' % sweep.name}
			threshold = -20
			
			feature = ephys.efeatures.eFELFeature(
				feature_name,
				efel_feature_name=efel_feature_name,
				recording_names=recording_names,
				stim_start=stim_start,
				stim_end=stim_end,
				exp_mean=mean,
				exp_std=sd,
				threshold=threshold,
				stimulus_current=stim_amp)
			
			fname = {}
			fname[efel_feature_name] = feature
			features_all.append(fname)
	
	# Create feature list for axonal features
	sweeps_axon = [sweeps[s-1] for s in s_axon] # Identify the sweeps where axonal features are evaluated
	for c, sweep in enumerate(sweeps_axon,c+1): # continues enumeration
		stim_start = sweep.stimuli[0].step_delay
		stim_end = stim_start + sweep.stimuli[0].step_duration
		stim_amp = sweep.stimuli[0].step_amplitude
		for efel_feature_name in feature_definitions[c]:
			feature_name = '%s.%s' % (sweep.name, efel_feature_name)
			mean = feature_definitions[c][efel_feature_name]
			sd = feature_sds[c][efel_feature_name]
			
			recording_names = {'': '%s.axon.v' % sweep.name}
			threshold = -20
			
			feature = ephys.efeatures.eFELFeature(
				feature_name,
				efel_feature_name=efel_feature_name,
				recording_names=recording_names,
				stim_start=stim_start,
				stim_end=stim_end,
				exp_mean=mean,
				exp_std=sd,
				threshold=threshold,
				stimulus_current=stim_amp)
			
			fname = {}
			fname[efel_feature_name] = feature
			features_all.append(fname)
	
	# Append all features of the same type to create list of max objective targets
	objectives = []
	AllFeatureDefinitions = set(x for l in feature_definitions for x in l)
	
	for efel_feature_name in AllFeatureDefinitions:
		tempfeatlist = [d[efel_feature_name] for d in features_all if efel_feature_name in d]
		if not tempfeatlist:
			continue
		objective = ephys.objectives.MaxObjective(
		efel_feature_name,
		tempfeatlist)
		objectives.append(objective)
	
	fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
	
	return fitcalc, objectives, AllFeatureDefinitions

fitness_calculator, objectives_to_save, featnames = define_fitness_calculator(sweep_protocols,all_features,all_features_sd)

# save for reconstructing ordering of errors later
f = open("results/objectives.pkl","wb")
pickle.dump(objectives_to_save,f,protocol=2)
f.close()
f = open("results/featurenames.pkl","wb")
pickle.dump(featnames,f,protocol=2)
f.close()

cell_evaluator = ephys.evaluators.CellEvaluator(
		cell_model=Cell_Model,
		param_names=nonfrozen_params,
		fitness_protocols={recording_protocol.name: recording_protocol},
		fitness_calculator=fitness_calculator,
		sim=nrn)
