######### Load Data for Active Optimization #########
Feature_DendDecay = 1
steps_dict = [20,23,35,37,39] # 140 pA (before rheobase), 150 pA (after rheobase), 250 pA
active_steps = [-0.11,-0.05,0.19,0.23,0.27] # nA
data_passive_dict = pickle.load(open("passive.pkl","rb"))
data_active_dict = pickle.load(open("active.pkl","rb"))
stimstart = 270
stimend = 1270

######### Get Active Features #########
## Largest hyperpolarization step
PassiveFeatures1 = ['sag_amplitude','decay_time_constant_after_stim','ohmic_input_resistance_vb_ssse','voltage_deflection','steady_state_hyper','voltage_base']

## Smallest hyperpolarization step
PassiveFeatures2 = ['sag_amplitude','decay_time_constant_after_stim','ohmic_input_resistance_vb_ssse','voltage_deflection','steady_state_hyper']

# Low spiking step
ActiveFeatures1 = ['Spikecount', 'AHP_depth_abs', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','inv_second_ISI','adaptation_index','AHP_depth_abs_slow', 'AHP_slow_time', 'AHP_time_from_peak']

# Mid spiking step
ActiveFeatures2 = ['Spikecount', 'AHP_depth_abs', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','inv_second_ISI','adaptation_index','AHP_depth_abs_slow', 'AHP_slow_time', 'AHP_time_from_peak']

# High spiking step
ActiveFeatures3 = ['Spikecount', 'AHP_depth_abs', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','inv_second_ISI','adaptation_index','AHP_depth_abs_slow', 'AHP_slow_time', 'AHP_time_from_peak']

# Constrain axonal spiking on highest step
ActiveFeatures4 = ['Spikecount', 'AHP_depth_abs', 'AP_height']

def get_active_features(data_pas,data_act):
	traces_pas1 = []
	traces_pas2 = []
	traces_act1 = []
	traces_act2 = []
	traces_act3 = []
	for step_name, step_traces in data_pas.items():
		trace = {}
		trace['T'] = data_pas[step_name]['T']
		trace['V'] = data_pas[step_name]['V']
		trace['stim_start'] = [stimstart]
		trace['stim_end'] = [stimend]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		if step_name == 0:
			traces_pas1.append(trace)
		elif step_name == 1:
			traces_pas2.append(trace)
	for step_name, step_traces in data_act.items():
		trace = {}
		trace['T'] = data_act[step_name]['T']
		trace['V'] = data_act[step_name]['V']
		trace['stim_start'] = [stimstart]
		trace['stim_end'] = [stimend]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		if step_name == 0:
			traces_act1.append(trace)
		elif step_name == 1:
			traces_act2.append(trace)
		elif step_name == 2:
			traces_act3.append(trace)
	
	features_values_pas1 = efel.getMeanFeatureValues(traces_pas1, PassiveFeatures1)
	features_values_pas2 = efel.getMeanFeatureValues(traces_pas2, PassiveFeatures2)
	features_values_act1 = efel.getMeanFeatureValues(traces_act1, ActiveFeatures1)
	features_values_act2 = efel.getMeanFeatureValues(traces_act2, ActiveFeatures2)
	features_values_act3 = efel.getMeanFeatureValues(traces_act3, ActiveFeatures3)
	
	features_values_act3_axon = {}
	features_values_act3_axon['AP_height'] = features_values_act3[0]['AP_height']
	features_values_act3_axon['AHP_depth_abs'] = features_values_act3[0]['AHP_depth_abs']
	features_values_act3_axon['Spikecount'] = features_values_act3[0]['Spikecount']
	
	features_values_pas1.extend(features_values_pas2)
	features_values_pas1.extend(features_values_act1)
	features_values_pas1.extend(features_values_act2)
	features_values_pas1.extend(features_values_act3)
	features_values_pas1.extend([features_values_act3_axon])
	return features_values_pas1

active_features = get_active_features(data_passive_dict,data_active_dict)

######### Create Mod Mechanism Dictionary #########
act_mechs={}
for x in active_params:
	if x['section'] == 'somatic':
		loc = somatic_loc
	elif x['section'] == 'basal':
		loc = basal_loc
	elif x['section'] == 'axonal':
		loc = axonal_loc
	elif x['section'] == 'all':
		loc = all_loc
	act_mechs["{0}{1}_mech".format(x['mechanism'],x['section'])]=ephys.mechanisms.NrnMODMechanism(
		name = x['mechanism']+x['section'],
		suffix = x['mechanism'],
		locations = [loc])

passomatic_mech = ephys.mechanisms.NrnMODMechanism(
	name ='passomatic',
	suffix ='pas',
	locations = [somatic_loc]
)
pasbasal_mech = ephys.mechanisms.NrnMODMechanism(
	name ='pasbasal',
	suffix ='pas',
	locations = [basal_loc]
)
pasaxonal_mech = ephys.mechanisms.NrnMODMechanism(
	name ='pasaxonal',
	suffix ='pas',
	locations = [axonal_loc]
)
act_mechs['passomatic_mech'] = passomatic_mech
act_mechs['pasbasal_mech'] = pasbasal_mech
act_mechs['pasaxonal_mech'] = pasaxonal_mech

mod_mechs = []
for name in act_mechs: mod_mechs.append(act_mechs[name])

######### Freeze Passive Parameters #########
celsius_param = ephys.parameters.NrnGlobalParameter(
	name = 'celsius',
	param_name ='celsius',
	value = 34,
	frozen = True
)
v_init_param = ephys.parameters.NrnGlobalParameter(
	name = 'v_init',
	param_name ='v_init',
	value = -81,
	frozen = True
)
Ra_param = ephys.parameters.NrnSectionParameter(
	name ='Ra',
	param_name ='Ra',
	locations = [all_loc],
	value = 100,
	frozen = True
)
cm_somatic_param = ephys.parameters.NrnSectionParameter(
	name ='cmsomatic',
	param_name ='cm',
	locations = [somatic_loc],
	value = 2,
	frozen = True
)
cm_basal_param = ephys.parameters.NrnSectionParameter(
	name ='cmbasal',
	param_name ='cm',
	locations = [basal_loc],
	value = 2,
	frozen = True
)
cm_axonal_param = ephys.parameters.NrnSectionParameter(
	name ='cmaxonal',
	param_name ='cm',
	locations = [axonal_loc],
	value = 2,
	frozen = True
)
g_pas_param = ephys.parameters.NrnSectionParameter(
	name ='g_pas',
	param_name ='g_pas',
	locations = [all_loc],
	value = 0.00026,
	bounds=[0.00022, 0.0004],
	frozen = False
)
e_pas_param = ephys.parameters.NrnSectionParameter(
	name ='e_pas',
	param_name ='e_pas',
	locations = [all_loc],
	value = -81,
	bounds=[-135, -73],
	frozen = False
)

######### Set Up Active Parameters to Optimize #########
act_params={}
for x in active_params:
	if x['section'] == 'somatic':
		loc = somatic_loc
	elif x['section'] == 'axonal':
		loc = axonal_loc
	elif x['section'] == 'all':
		loc = all_loc
	if x['name'] == 'decay_CaDynamics':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 80,
			bounds=[20, 1000],
			frozen = False)
	elif x['name'] == 'gamma_CaDynamics':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.0005,
			frozen = True)
	elif x['name'] == 'gbar_NaTg':
		if x['section'] == 'somatic':
			act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
				name =x['name']+x['section'],
				param_name =x['name'],
				locations = [loc],
				value = 0.05,
				bounds=[0, 0.5],
				frozen = False)
			act_params["{0}{1}_param".format('vshiftm_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='vshiftm_NaTg'+x['section'],
				param_name ='vshiftm_NaTg',
				locations = [loc],
				value = 0,
				frozen = True)
			act_params["{0}{1}_param".format('vshifth_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='vshifth_NaTg'+x['section'],
				param_name ='vshifth_NaTg',
				locations = [loc],
				value = 10,
				frozen = True)
			act_params["{0}{1}_param".format('slopem_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='slopem_NaTg'+x['section'],
				param_name ='slopem_NaTg',
				locations = [loc],
				value = 9,
				frozen = True)
		elif x['section'] == 'axonal':
			act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
				name =x['name']+x['section'],
				param_name =x['name'],
				locations = [loc],
				value = 0.15,
				bounds=[0, 0.7],
				frozen = False)
			act_params["{0}{1}_param".format('vshifth_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='vshifth_NaTg'+x['section'],
				param_name ='vshifth_NaTg',
				locations = [loc],
				value = 10,
				frozen = True)
			act_params["{0}{1}_param".format('slopem_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='slopem_NaTg'+x['section'],
				param_name ='slopem_NaTg',
				locations = [loc],
				value = 9,
				frozen = True)
	elif x['name'] == 'gbar_Nap':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-3,
			bounds=[0, 1e-1],
			frozen = False)
	elif x['name'] == 'gbar_K_P':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.1,
			bounds=[0, 1],
			frozen = False)
	elif x['name'] == 'gbar_K_T':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-2,
			bounds=[0, 1e-1],
			frozen = False)
	elif x['name'] == 'gbar_SK':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-4,
			bounds=[0, 1],
			frozen = False)
	elif x['name'] == 'gbar_Kv3_1':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.6,
			bounds=[0, 3],
			frozen = False)
		act_params["{0}{1}_param".format('vshift_Kv3_1',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='vshift_Kv3_1'+x['section'],
			param_name ='vshift_Kv3_1',
			locations = [loc],
			value = 0,
			frozen = True)
	elif x['name'] == 'gbar_Ca_HVA':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-4,
			bounds=[0, 1e-2],
			frozen = False)
	elif x['name'] == 'gbar_Ca_LVA':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-4,
			bounds=[0, 1e-1],
			frozen = False)
	elif x['name'] == 'gbar_Im':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-5,
			bounds=[0, 1e-1],
			frozen = False)
	elif (x['name'] == 'gbar_Ih') and (x['section'] == 'all'):
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0,
			bounds = [0, 0.001],
			frozen = False)
		act_params["{0}{1}_param".format('shift1_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift1_Ih'+x['section'],
			param_name ='shift1_Ih',
			locations = [loc],
			value = 262.07471272583734,
			frozen = True)
		act_params["{0}{1}_param".format('shift2_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift2_Ih'+x['section'],
			param_name ='shift2_Ih',
			locations = [loc],
			value = 19.041504011554068,
			frozen = True)
		act_params["{0}{1}_param".format('shift3_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift3_Ih'+x['section'],
			param_name ='shift3_Ih',
			locations = [loc],
			value = 4.162478599311996,
			frozen = True)
		act_params["{0}{1}_param".format('shift4_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift4_Ih'+x['section'],
			param_name ='shift4_Ih',
			locations = [loc],
			value = 75.56131984386032,
			frozen = True)
		act_params["{0}{1}_param".format('shift5_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift5_Ih'+x['section'],
			param_name ='shift5_Ih',
			locations = [loc],
			value = 78.52193065082363,
			frozen = True)
		act_params["{0}{1}_param".format('shift6_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift6_Ih'+x['section'],
			param_name ='shift6_Ih',
			locations = [loc],
			value = 1.3115816183914442,
			frozen = True)
	else:
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.5,
			bounds=[0, 1],
			frozen = False)

ena_paramsomatic = ephys.parameters.NrnSectionParameter(
	name ='enasomatic',
	param_name ='ena',
	locations = [somatic_loc],
	value = 50,
	frozen = True
)
ena_paramaxonal = ephys.parameters.NrnSectionParameter(
	name ='enaaxonal',
	param_name ='ena',
	locations = [axonal_loc],
	value = 50,
	frozen = True
)
ek_paramsomatic = ephys.parameters.NrnSectionParameter(
	name ='eksomatic',
	param_name ='ek',
	locations = [somatic_loc],
	value = -85,
	frozen = True
)
ek_paramaxonal = ephys.parameters.NrnSectionParameter(
	name ='ekaxonal',
	param_name ='ek',
	locations = [axonal_loc],
	value = -85,
	frozen = True
)

mod_params = [celsius_param, v_init_param, ena_paramsomatic, ena_paramaxonal, ek_paramsomatic, ek_paramaxonal, e_pas_param, Ra_param, cm_somatic_param, cm_basal_param, cm_axonal_param, g_pas_param]
for name in act_params: mod_params.append(act_params[name])

######### Create Model #########

Cell_Model = ephys.models.CellModel(
	name = cellname,
	morph = morphology,
	mechs = mod_mechs,
	params = mod_params
)

######### Choose Recording Sites #########
soma_loc = ephys.locations.NrnSeclistCompLocation(
	name='soma',
	seclist_name='somatic',
	sec_index = 0,
	comp_x = 0.5
)
# Assuming 2 sections away - this approach is flawed because of variability in morphs
axon_loc = ephys.locations.NrnSeclistCompLocation(
	name='axon',
	seclist_name='axonal',
	sec_index = 0,
	comp_x = 0.5
)
nrn = ephys.simulators.NrnSimulator()

h.tstop = 2000
h.dt = 0.025

######### Create Stimuli for Active Optimizations #########
sweep_protocols_act = []
stim_protocol = []
rec_protocol = []
for protocol_name, amplitude in [('step1', active_steps[0]), ('step2', active_steps[1]), ('step3', active_steps[2]), ('step4', active_steps[3]), ('step5', active_steps[4])]:
	# Set-up current-clamp stimuli
	stim = ephys.stimuli.NrnSquarePulse(
				step_amplitude=amplitude,
				step_delay=stimstart,
				step_duration=stimend-stimstart,
				location=soma_loc,
				total_duration=h.tstop)
	rec = ephys.recordings.CompRecording(
			name='%s.soma.v' % protocol_name,
			location=soma_loc,
			variable='v')
	rec2 = ephys.recordings.CompRecording(
			name='%s.axon.v' % protocol_name,
			location=axon_loc,
			variable='v')
	protocol = ephys.protocols.SweepProtocol(name=protocol_name, stimuli=[stim], recordings=[rec,rec2], cvode_active=1)
	stim_protocol.append(stim)
	rec_protocol.append(rec)
	sweep_protocols_act.append(protocol)

active_fivestep_protocol = ephys.protocols.SequenceProtocol(name='fivestep', protocols=sweep_protocols_act)

#################################
### Set-up Fitness Calculator ###
#################################

def define_fitness_calculator(sweeps,feature_definitions):
	"""Define fitness calculator"""
	
	objectives = []
	features_pas = []
	features_act = []
	stdvec = []
	c = 0
	for sweep in sweeps:
		stim_start = sweep.stimuli[0].step_delay
		stim_end = stim_start + sweep.stimuli[0].step_duration
		stim_amp = sweep.stimuli[0].step_amplitude
		for efel_feature_name in feature_definitions[c]:
			feature_name = '%s.%s' % (sweep.name, efel_feature_name)
			mean = feature_definitions[c][efel_feature_name]
			if mean == 0:
				std = 0.001
			elif efel_feature_name == 'ohmic_input_resistance_vb_ssse':
				if sweep.stimuli[0].step_amplitude == active_steps[0]:
					std = 0.1
				if sweep.stimuli[0].step_amplitude == active_steps[1]:
					std = 0.1
			elif efel_feature_name == 'sag_amplitude':
				std = 0.1
				if sweep.stimuli[0].step_amplitude == active_steps[1]:
					# print('Resetting sag_amplitude mean at ' + str(sweep.stimuli[0].step_amplitude) + " nA to 0")
					# mean = 0 # since sag on smaller step should hopefully be zero mV for passive fit
					std = 0.1 # since sag on smaller step should hopefully be zero mV for passive fit
			elif efel_feature_name == 'voltage_base':
				std = 0.1
			elif efel_feature_name == 'decay_time_constant_after_stim':
				std = 0.1
			elif efel_feature_name == 'voltage_deflection':
				std = 0.1
			elif efel_feature_name == 'steady_state_hyper':
				std = 0.1
			elif efel_feature_name == 'Spikecount':
				std = 0.05
			elif efel_feature_name == 'AHP_depth_abs':
				std = 1
			elif efel_feature_name == 'AP_width':
				std = 0.05
			elif efel_feature_name == 'AP_height':
				std = 4
			elif efel_feature_name == 'ISI_CV':
				std = 0.01
			elif efel_feature_name == 'inv_first_ISI':
				std = 1
			elif efel_feature_name == 'inv_second_ISI':
				std = 1
			elif efel_feature_name == 'AHP_depth_abs_slow':
				std = 0.1
			elif efel_feature_name == 'AHP_slow_time':
				std = 0.05
			elif efel_feature_name == 'AHP_time_from_peak':
				std = 0.03
			elif efel_feature_name == 'adaptation_index':
				std = 0.0001
			else:
				std = abs(mean*0.1)
			stdvec.append(std)
			
			recording_names = {'': '%s.soma.v' % sweep.name}
			threshold = -20
			
			feature = ephys.efeatures.eFELFeature(
				feature_name,
				efel_feature_name=efel_feature_name,
				recording_names=recording_names,
				stim_start=stim_start,
				stim_end=stim_end,
				exp_mean=mean,
				exp_std=std,
				threshold=threshold)
			
			fname = {}
			fname[efel_feature_name] = feature
			if stim_amp < 0:
				features_pas.append(fname)
			elif stim_amp > 0:
				features_act.append(fname)
			
		c = c + 1
		
	sweep = sweeps[4]
	stim_start = sweep.stimuli[0].step_delay
	stim_end = stim_start + sweep.stimuli[0].step_duration
	for efel_feature_name in feature_definitions[c]:
		feature_name = '%s.%s' % (sweep.name, efel_feature_name)
		mean = feature_definitions[c][efel_feature_name]
		if mean == 0:
			std = 0.001
		elif efel_feature_name == 'Spikecount':
			std = 0.05
		elif efel_feature_name == 'AHP_depth_abs':
			std = 0.5
		elif efel_feature_name == 'AP_width':
			std = 0.05
		elif efel_feature_name == 'AP_height':
			std = 2
		elif efel_feature_name == 'ISI_CV':
			std = 0.01
		else:
			std = abs(mean*0.1)
		stdvec.append(std)
		
		recording_names = {'': '%s.axon.v' % sweep.name}
		threshold = -20
		
		feature = ephys.efeatures.eFELFeature(
			feature_name,
			efel_feature_name=efel_feature_name,
			recording_names=recording_names,
			stim_start=stim_start,
			stim_end=stim_end,
			exp_mean=mean,
			exp_std=std,
			threshold=threshold)
			
		fname = {}
		fname[efel_feature_name] = feature
		features_act.append(fname)
	
	# Passive objectives
	AllFeatureDefinitions = set(x for l in feature_definitions for x in l)
	
	for efel_feature_name in AllFeatureDefinitions:
		tempfeatlist = [d[efel_feature_name] for d in features_pas if efel_feature_name in d]
		if not tempfeatlist:
			continue
		objective = ephys.objectives.MaxObjective(
		efel_feature_name,
		tempfeatlist)
		objectives.append(objective)
	
	# Active objectives
	for efel_feature_name in AllFeatureDefinitions:
		tempfeatlist = [d[efel_feature_name] for d in features_act if efel_feature_name in d]
		if not tempfeatlist:
			continue
		objective = ephys.objectives.MaxObjective(
		efel_feature_name,
		tempfeatlist)
		objectives.append(objective)
	
	fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
	
	return fitcalc, stdvec

fitness_calculator_act, stdvec_act = define_fitness_calculator(sweep_protocols_act,active_features)

nonfrozen_params_act=['g_pas','e_pas']
for x in active_params:
	if (x['name'] =='gamma_CaDynamics'):
		continue
	else:
		nonfrozen_params_act.append(x['name']+x['section'])

# nonfrozen_params_act = set(nonfrozen_params_act)

cell_evaluator_act = ephys.evaluators.CellEvaluator(
		cell_model=Cell_Model,
		param_names=nonfrozen_params_act,
		fitness_protocols={active_fivestep_protocol.name: active_fivestep_protocol},
		fitness_calculator=fitness_calculator_act,
		sim=nrn)

##############################
### Setup Parallel Context ###
##############################
from ipyparallel import Client
rc = Client(profile=os.getenv('IPYTHON_PROFILE'))

sys.stderr.write('Using ipyparallel with {0} engines'.format(len(rc)))
sys.stderr.flush()
sys.stderr.write("\n")
sys.stderr.flush()

lview = rc.load_balanced_view()

def mapper(func, it):
	start_time = time.time()
	ret = lview.map_async(func, it)
	sys.stderr.write('Generation took {0}'.format(time.time() - start_time))
	sys.stderr.flush()
	sys.stderr.write("\n")
	sys.stderr.flush()
	return ret

map_function = mapper

########################
### Run Optimization ###
########################
offss = 400
ngens = 300
SELECTOR = 'IBEA' # IBEA or NSGA2

# Note: Set use_scoop and isolate to True when moving to parallelized context
optimisation = bpop.optimisations.DEAPOptimisation(
		evaluator=cell_evaluator_act,
		map_function = map_function,
		cxpb = 0.1,
		mutpb = 0.35,
		eta = 10,
		offspring_size = offss,
		selector_name = SELECTOR,
		seed=8619)

finalpop, halloffame, logs, hist = optimisation.run(max_ngen=ngens)

##### Save Population #####
f = open("results/finalpop_act.pkl","wb")
pickle.dump(finalpop,f,protocol=2)
f.close()
f = open("results/halloffame_act.pkl","wb")
pickle.dump(halloffame,f,protocol=2)
f.close()
f = open("results/logs_act.pkl","wb")
pickle.dump(logs,f,protocol=2)
f.close()
f = open("results/hist_act.pkl","wb")
pickle.dump(hist,f,protocol=2)
f.close()

##### Print Top Models #####
print('Final population: \n', finalpop, "\n")
sys.stdout.flush()
best_ind = halloffame[0]
best_ind2 = halloffame[1]
best_ind3 = halloffame[2]
best_ind4 = halloffame[3]
best_ind5 = halloffame[4]

print(nonfrozen_params_act)
sys.stdout.flush()
print('Best individual: ', list(zip(nonfrozen_params_act,best_ind)))
sys.stdout.flush()
print('2nd individual: ', list(zip(nonfrozen_params_act,best_ind2)))
sys.stdout.flush()
print('3rd individual: ', list(zip(nonfrozen_params_act,best_ind3)))
sys.stdout.flush()
print('4th individual: ', list(zip(nonfrozen_params_act,best_ind4)))
sys.stdout.flush()
print('5th individual: ', list(zip(nonfrozen_params_act,best_ind5)))
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

best_ind_dict = cell_evaluator_act.param_dict(best_ind)
best_ind_dict2 = cell_evaluator_act.param_dict(best_ind2)
best_ind_dict3 = cell_evaluator_act.param_dict(best_ind3)
best_ind_dict4 = cell_evaluator_act.param_dict(best_ind4)
best_ind_dict5 = cell_evaluator_act.param_dict(best_ind5)

##### Plot Top Models #####
def plot_responses_pas(responses):
	ind1 = numpy.where(data_passive_dict[0]['T'] == 270)
	ind2 = numpy.where(data_passive_dict[1]['T'] == 270)
	shift1 = 0
	shift2 = data_passive_dict[1]['V'][ind2]-data_passive_dict[0]['V'][ind1]
	
	plt.subplot(2,1,1)
	plt.plot(data_passive_dict[0]['T'],data_passive_dict[0]['V']-shift1, color='b')
	plt.plot(responses['step1.soma.v']['time'], responses['step1.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-100,-80)
	plt.subplot(2,1,2)
	plt.plot(data_passive_dict[1]['T'],data_passive_dict[1]['V']-shift2, color='b')
	plt.plot(responses['step2.soma.v']['time'], responses['step2.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-100,-80)
	plt.tight_layout()

def plot_responses_act(responses):
	plt.subplot(3,1,1)
	plt.plot(data_active_dict[0]['T'],data_active_dict[0]['V'], color='b')
	plt.plot(responses['step3.soma.v']['time'], responses['step3.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.subplot(3,1,2)
	plt.plot(data_active_dict[1]['T'],data_active_dict[1]['V'], color='b')
	plt.plot(responses['step4.soma.v']['time'], responses['step4.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.subplot(3,1,3)
	plt.plot(data_active_dict[2]['T'],data_active_dict[2]['V'], color='b')
	plt.plot(responses['step5.soma.v']['time'], responses['step5.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.tight_layout()

def plot_responses_act2(responses):
	plt.subplot(3,1,1)
	plt.plot(responses['step3.axon.v']['time'], responses['step3.axon.v']['voltage'], color='m')
	plt.plot(responses['step3.soma.v']['time'], responses['step3.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.subplot(3,1,2)
	plt.plot(responses['step4.axon.v']['time'], responses['step4.axon.v']['voltage'], color='m')
	plt.plot(responses['step4.soma.v']['time'], responses['step4.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.subplot(3,1,3)
	plt.plot(responses['step5.axon.v']['time'], responses['step5.axon.v']['voltage'], color='m')
	plt.plot(responses['step5.soma.v']['time'], responses['step5.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.tight_layout()

active_steps = [-0.11,-0.05,0.17,0.21,0.25,0.25] # nA
def get_active_features2(data):
	traces_pas1 = []
	traces_pas2 = []
	traces_act1 = []
	traces_act2 = []
	traces_act3 = []
	traces_act4 = []
	for step_name, step_traces in data.items():
		trace = {}
		trace['T'] = data[step_name]['T']
		trace['V'] = data[step_name]['V']
		trace['stim_start'] = [270]
		trace['stim_end'] = [1270]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		if step_name == 0:
			traces_pas1.append(trace)
		elif step_name == 1:
			traces_pas2.append(trace)
		elif step_name == 2:
			traces_act1.append(trace)
		elif step_name == 3:
			traces_act2.append(trace)
		elif step_name == 4:
			traces_act3.append(trace)
		elif step_name == 5:
			traces_act4.append(trace)
	
	features_values_pas1 = efel.getMeanFeatureValues(traces_pas1, PassiveFeatures1)
	features_values_pas2 = efel.getMeanFeatureValues(traces_pas2, PassiveFeatures2)
	features_values_act1 = efel.getMeanFeatureValues(traces_act1, ActiveFeatures1)
	features_values_act2 = efel.getMeanFeatureValues(traces_act2, ActiveFeatures2)
	features_values_act3 = efel.getMeanFeatureValues(traces_act3, ActiveFeatures3)
	features_values_act4 = efel.getMeanFeatureValues(traces_act4, ActiveFeatures4)
	
	features_values_pas1.extend(features_values_pas2)
	features_values_pas1.extend(features_values_act1)
	features_values_pas1.extend(features_values_act2)
	features_values_pas1.extend(features_values_act3)
	features_values_pas1.extend(features_values_act4)
	return features_values_pas1

def get_stdevs(data):
	num_stdevs = []
	counter = 0
	for t in range(len(active_features)):
		err={}
		for feat in active_features[t]:
			if data[t][feat] is not None:
				err[feat] = abs((data[t][feat]-active_features[t][feat])/stdvec_act[counter])
			else:
				err[feat] = None
			
			counter = counter + 1
		num_stdevs.append(err)
	
	return num_stdevs

responses = active_fivestep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict, sim=nrn)
plot_responses_pas(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_pas.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_pas.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_act2.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces1_act2.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

data = collections.OrderedDict()
sweep_data1 = {}
sweep_data1['V'] = responses['step1.soma.v']['voltage']
sweep_data1['T'] = responses['step1.soma.v']['time']
sweep_data2 = {}
sweep_data2['V'] = responses['step2.soma.v']['voltage']
sweep_data2['T'] = responses['step2.soma.v']['time']
sweep_data3 = {}
sweep_data3['V'] = responses['step3.soma.v']['voltage']
sweep_data3['T'] = responses['step3.soma.v']['time']
sweep_data4 = {}
sweep_data4['V'] = responses['step4.soma.v']['voltage']
sweep_data4['T'] = responses['step4.soma.v']['time']
sweep_data5 = {}
sweep_data5['V'] = responses['step5.soma.v']['voltage']
sweep_data5['T'] = responses['step5.soma.v']['time']
sweep_data6 = {}
sweep_data6['V'] = responses['step5.axon.v']['voltage']
sweep_data6['T'] = responses['step5.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
active_features1 = get_active_features2(data)

print("\nTop Model 1 Features:\n")
sys.stdout.flush()
pp.pprint(active_features1)
sys.stdout.flush()

active_errs1 = get_stdevs(active_features1)

print("\nTop Model 1 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs1)
sys.stdout.flush()

responses = active_fivestep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict2, sim=nrn)
plot_responses_pas(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_pas.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_pas.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_act2.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces2_act2.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

data = collections.OrderedDict()
sweep_data1 = {}
sweep_data1['V'] = responses['step1.soma.v']['voltage']
sweep_data1['T'] = responses['step1.soma.v']['time']
sweep_data2 = {}
sweep_data2['V'] = responses['step2.soma.v']['voltage']
sweep_data2['T'] = responses['step2.soma.v']['time']
sweep_data3 = {}
sweep_data3['V'] = responses['step3.soma.v']['voltage']
sweep_data3['T'] = responses['step3.soma.v']['time']
sweep_data4 = {}
sweep_data4['V'] = responses['step4.soma.v']['voltage']
sweep_data4['T'] = responses['step4.soma.v']['time']
sweep_data5 = {}
sweep_data5['V'] = responses['step5.soma.v']['voltage']
sweep_data5['T'] = responses['step5.soma.v']['time']
sweep_data6 = {}
sweep_data6['V'] = responses['step5.axon.v']['voltage']
sweep_data6['T'] = responses['step5.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
active_features2 = get_active_features2(data)

print("\nTop Model 2 Features:\n")
sys.stdout.flush()
pp.pprint(active_features2)
sys.stdout.flush()

active_errs2 = get_stdevs(active_features2)

print("\nTop Model 2 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs2)
sys.stdout.flush()

responses = active_fivestep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict3, sim=nrn)
plot_responses_pas(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_pas.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_pas.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_act2.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces3_act2.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

data = collections.OrderedDict()
sweep_data1 = {}
sweep_data1['V'] = responses['step1.soma.v']['voltage']
sweep_data1['T'] = responses['step1.soma.v']['time']
sweep_data2 = {}
sweep_data2['V'] = responses['step2.soma.v']['voltage']
sweep_data2['T'] = responses['step2.soma.v']['time']
sweep_data3 = {}
sweep_data3['V'] = responses['step3.soma.v']['voltage']
sweep_data3['T'] = responses['step3.soma.v']['time']
sweep_data4 = {}
sweep_data4['V'] = responses['step4.soma.v']['voltage']
sweep_data4['T'] = responses['step4.soma.v']['time']
sweep_data5 = {}
sweep_data5['V'] = responses['step5.soma.v']['voltage']
sweep_data5['T'] = responses['step5.soma.v']['time']
sweep_data6 = {}
sweep_data6['V'] = responses['step5.axon.v']['voltage']
sweep_data6['T'] = responses['step5.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
active_features3 = get_active_features2(data)

print("\nTop Model 3 Features:\n")
sys.stdout.flush()
pp.pprint(active_features3)
sys.stdout.flush()

active_errs3 = get_stdevs(active_features3)

print("\nTop Model 3 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs3)
sys.stdout.flush()

responses = active_fivestep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict4, sim=nrn)
plot_responses_pas(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_pas.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_pas.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_act2.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces4_act2.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

data = collections.OrderedDict()
sweep_data1 = {}
sweep_data1['V'] = responses['step1.soma.v']['voltage']
sweep_data1['T'] = responses['step1.soma.v']['time']
sweep_data2 = {}
sweep_data2['V'] = responses['step2.soma.v']['voltage']
sweep_data2['T'] = responses['step2.soma.v']['time']
sweep_data3 = {}
sweep_data3['V'] = responses['step3.soma.v']['voltage']
sweep_data3['T'] = responses['step3.soma.v']['time']
sweep_data4 = {}
sweep_data4['V'] = responses['step4.soma.v']['voltage']
sweep_data4['T'] = responses['step4.soma.v']['time']
sweep_data5 = {}
sweep_data5['V'] = responses['step5.soma.v']['voltage']
sweep_data5['T'] = responses['step5.soma.v']['time']
sweep_data6 = {}
sweep_data6['V'] = responses['step5.axon.v']['voltage']
sweep_data6['T'] = responses['step5.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
active_features4 = get_active_features2(data)

print("\nTop Model 4 Features:\n")
sys.stdout.flush()
pp.pprint(active_features4)
sys.stdout.flush()

active_errs4 = get_stdevs(active_features4)

print("\nTop Model 4 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs4)
sys.stdout.flush()

responses = active_fivestep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict5, sim=nrn)
plot_responses_pas(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_pas.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_pas.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_act2.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_OptimizedTraces5_act2.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()

data = collections.OrderedDict()
sweep_data1 = {}
sweep_data1['V'] = responses['step1.soma.v']['voltage']
sweep_data1['T'] = responses['step1.soma.v']['time']
sweep_data2 = {}
sweep_data2['V'] = responses['step2.soma.v']['voltage']
sweep_data2['T'] = responses['step2.soma.v']['time']
sweep_data3 = {}
sweep_data3['V'] = responses['step3.soma.v']['voltage']
sweep_data3['T'] = responses['step3.soma.v']['time']
sweep_data4 = {}
sweep_data4['V'] = responses['step4.soma.v']['voltage']
sweep_data4['T'] = responses['step4.soma.v']['time']
sweep_data5 = {}
sweep_data5['V'] = responses['step5.soma.v']['voltage']
sweep_data5['T'] = responses['step5.soma.v']['time']
sweep_data6 = {}
sweep_data6['V'] = responses['step5.axon.v']['voltage']
sweep_data6['T'] = responses['step5.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
active_features5 = get_active_features2(data)

print("\nTop Model 5 Features:\n")
sys.stdout.flush()
pp.pprint(active_features5)
sys.stdout.flush()

active_errs5 = get_stdevs(active_features5)

print("\nTop Model 5 Errors:\n")
sys.stdout.flush()
pp.pprint(active_errs5)
sys.stdout.flush()

print("\nExperimental Features:\n")
sys.stdout.flush()
pp.pprint(active_features)
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
plt.savefig('PLOTfiles/' + cellname + '_Performance_act.pdf', bbox_inches='tight')
plt.savefig('PLOTfiles/' + cellname + '_Performance_act.png', bbox_inches='tight')
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
