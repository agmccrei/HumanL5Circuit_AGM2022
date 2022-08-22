######### Load Data for Active Optimization #########
active_steps = [-0.4,-0.3,-0.2,-0.05,0.1,0.2,0.3] # nA
data_active_dict = pickle.load(open("active.pkl","rb"))
stimstart = 1000
stimend = 1600
# best_ind = [0.00013276715388485768,-83.85905159083714,4.0286355749662496e-05]

######### Get Active Features #########
# For 3 hyperpolarization steps
PassiveFeatures1 = ['sag_amplitude','sag_ratio1','ohmic_input_resistance_vb_ssse','voltage_base']
PassiveFeatures2 = ['sag_amplitude','sag_ratio1','ohmic_input_resistance_vb_ssse','voltage_base']
PassiveFeatures3 = ['sag_amplitude','sag_ratio1','ohmic_input_resistance_vb_ssse','voltage_base']
PassiveFeatures4 = ['sag_amplitude','sag_ratio1','ohmic_input_resistance_vb_ssse','voltage_base']

# Low spiking step
ActiveFeatures1 = ['Spikecount', 'mean_frequency', 'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_slow_time', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','AP_fall_time','steady_state_voltage_stimend','decay_time_constant_after_stim']

# Mid spiking step
ActiveFeatures2 = ['Spikecount', 'mean_frequency', 'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_slow_time', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','AP_fall_time','steady_state_voltage_stimend','decay_time_constant_after_stim']

# High spiking step
ActiveFeatures3 = ['Spikecount', 'mean_frequency', 'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_slow_time', 'AP_width', 'AP_height','ISI_CV','inv_first_ISI','AP_fall_time','steady_state_voltage_stimend','decay_time_constant_after_stim']

# Constrain axonal spiking on spike steps
ActiveFeatures4 = ['Spikecount', 'mean_frequency','steady_state_voltage_stimend','decay_time_constant_after_stim']
ActiveFeatures5 = ['Spikecount', 'mean_frequency','steady_state_voltage_stimend','decay_time_constant_after_stim']
ActiveFeatures6 = ['Spikecount', 'mean_frequency','steady_state_voltage_stimend','decay_time_constant_after_stim']

def get_active_features(data,manual):
	traces_pas1 = []
	traces_pas2 = []
	traces_pas3 = []
	traces_pas4 = []
	traces_act1 = []
	traces_act2 = []
	traces_act3 = []
	for step_name, step_traces in data.items():
		trace = {}
		trace['T'] = data[step_name]['T']
		trace['V'] = data[step_name]['V']
		trace['stim_start'] = [stimstart]
		trace['stim_end'] = [stimend]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		if step_name == 0:
			traces_pas1.append(trace)
		elif step_name == 1:
			traces_pas2.append(trace)
		elif step_name == 2:
			traces_pas3.append(trace)
		elif step_name == 3:
			traces_pas4.append(trace)
		elif step_name == 4:
			traces_act1.append(trace)
		elif step_name == 5:
			traces_act2.append(trace)
		elif step_name == 6:
			traces_act3.append(trace)
	
	if manual == 1:
		features_values1 = numpy.array([dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([])])
		features_values2 = numpy.array([dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([]),dict([])])
		
		features_values1[0]['sag_amplitude'] = 3.6775989649854615
		features_values1[0]['sag_ratio1'] = 0.12221060716408808
		features_values1[0]['ohmic_input_resistance_vb_ssse'] = 72.28615724524
		features_values1[0]['voltage_base'] = -71.87957306165225
		
		features_values2[0]['sag_amplitude_std'] = 1.8254970526175351/10
		features_values2[0]['sag_ratio1_std'] = 0.05010451751355496
		features_values2[0]['ohmic_input_resistance_vb_ssse_std'] = 33.454297498857684
		features_values2[0]['voltage_base_std'] = 5.007265435844746/2
		
		features_values1[1]['sag_amplitude'] = 2.927076006742862
		features_values1[1]['sag_ratio1'] = 0.11930780081786073
		features_values1[1]['ohmic_input_resistance_vb_ssse'] = 79.08759551159801
		features_values1[1]['voltage_base'] = -72.1908270945765
		
		features_values2[1]['sag_amplitude_std'] = 1.5823616224205286/10
		features_values2[1]['sag_ratio1_std'] = 0.05582615959936093
		features_values2[1]['ohmic_input_resistance_vb_ssse_std'] = 35.82516245212583
		features_values2[1]['voltage_base_std'] = 5.147306955246066/2
		
		features_values1[2]['sag_amplitude'] = 2.1625007279520525
		features_values1[2]['sag_ratio1'] = 0.11856984168498061
		features_values1[2]['ohmic_input_resistance_vb_ssse'] = 88.07095848954897
		features_values1[2]['voltage_base'] = -72.23412015231611
		
		features_values2[2]['sag_amplitude_std'] = 1.2417425256840757/10
		features_values2[2]['sag_ratio1_std'] = 0.056727669541529774
		features_values2[2]['ohmic_input_resistance_vb_ssse_std'] = 41.148312874114666
		features_values2[2]['voltage_base_std'] = 5.106214743866047/2
		
		features_values1[3]['sag_amplitude'] = 0.9321999814343563
		features_values1[3]['sag_ratio1'] = 0.16311846985029396
		features_values1[3]['ohmic_input_resistance_vb_ssse'] = 113.62408571001566
		features_values1[3]['voltage_base'] = -72.52907339557296
		
		features_values2[3]['sag_amplitude_std'] = 0.5493372552313776/5
		features_values2[3]['sag_ratio1_std'] = 0.08759962622859466
		features_values2[3]['ohmic_input_resistance_vb_ssse_std'] = 64.55495760776186
		features_values2[3]['voltage_base_std'] = 5.106458376803106/2
		
		features_values1[4]['Spikecount'] = 3.369565217391304
		features_values1[4]['mean_frequency'] = 11.458810961754287
		features_values1[4]['AHP_depth_abs'] = -57.521739838519466
		features_values1[4]['AHP_depth_abs_slow'] = -60.390207113306964
		features_values1[4]['AHP_slow_time'] = 0.3333859478768642
		features_values1[4]['AP_width'] = 2.7579438666496285
		features_values1[4]['AP_height'] = 40.56392265613454
		features_values1[4]['ISI_CV'] = 0.22498975418266395
		features_values1[4]['inv_first_ISI'] = 15.476516047035451
		features_values1[4]['AP_fall_time'] = 1.2651668519973964
		features_values1[4]['steady_state_voltage_stimend'] = -60.85488734585755
		features_values1[4]['decay_time_constant_after_stim'] = 21.31540362874316
		
		features_values2[4]['Spikecount_std'] = 4.034410121486831
		features_values2[4]['mean_frequency_std'] = 5.701649506179596
		features_values2[4]['AHP_depth_abs_std'] = 9.086467319757858
		features_values2[4]['AHP_depth_abs_slow_std'] = 8.433515069032662
		features_values2[4]['AHP_slow_time_std'] = 0.17819762253430566
		features_values2[4]['AP_width_std'] = 1.1722186708457936
		features_values2[4]['AP_height_std'] = 18.232267569594782
		features_values2[4]['ISI_CV_std'] = 0.177467811369984
		features_values2[4]['inv_first_ISI_std'] = 21.149583420553263
		features_values2[4]['AP_fall_time_std'] = 1.08964117551609
		features_values2[4]['steady_state_voltage_stimend_std'] = 7.3007534171312445
		features_values2[4]['decay_time_constant_after_stim_std'] = 11.470512186536931
		
		features_values1[5]['Spikecount'] = 8.021739130434783
		features_values1[5]['mean_frequency'] = 17.47538320358037
		features_values1[5]['AHP_depth_abs'] = -55.11718163812068
		features_values1[5]['AHP_depth_abs_slow'] = -56.94104874515491
		features_values1[5]['AHP_slow_time'] = 0.342579703651345
		features_values1[5]['AP_width'] = 3.097653647335267
		features_values1[5]['AP_height'] = 38.18447403704835
		features_values1[5]['ISI_CV'] = 0.17219712799819584
		features_values1[5]['inv_first_ISI'] = 35.12483094526111
		features_values1[5]['AP_fall_time'] = 1.3140592070249946
		features_values1[5]['steady_state_voltage_stimend'] = -54.003600639556545
		features_values1[5]['decay_time_constant_after_stim'] = 20.162479596388913
		
		features_values2[5]['Spikecount_std'] = 6.091651915727116/2
		features_values2[5]['mean_frequency_std'] = 8.418968352967584
		features_values2[5]['AHP_depth_abs_std'] = 9.079669174154857
		features_values2[5]['AHP_depth_abs_slow_std'] = 8.426339815209124
		features_values2[5]['AHP_slow_time_std'] = 0.17935361059261942
		features_values2[5]['AP_width_std'] = 1.697045648388051
		features_values2[5]['AP_height_std'] = 17.381636370330813
		features_values2[5]['ISI_CV_std'] = 0.09582519792723977
		features_values2[5]['inv_first_ISI_std'] = 30.369455768751045
		features_values2[5]['AP_fall_time_std'] = 1.0423072951947636
		features_values2[5]['steady_state_voltage_stimend_std'] = 8.392072647609883
		features_values2[5]['decay_time_constant_after_stim_std'] = 23.008748644471066
		
		features_values1[6]['Spikecount'] = 11.391304347826088
		features_values1[6]['mean_frequency'] = 21.551579074071693
		features_values1[6]['AHP_depth_abs'] = -51.83975560104322
		features_values1[6]['AHP_depth_abs_slow'] = -52.88779249387303
		features_values1[6]['AHP_slow_time'] = 0.37530612717992234
		features_values1[6]['AP_width'] = 4.138398264273764
		features_values1[6]['AP_height'] = 32.55730987302576
		features_values1[6]['ISI_CV'] = 0.26140057973597586
		features_values1[6]['inv_first_ISI'] = 52.22021020447993
		features_values1[6]['AP_fall_time'] = 1.1636839349052464
		features_values1[6]['steady_state_voltage_stimend'] = -49.04723567170013
		features_values1[6]['decay_time_constant_after_stim'] = 14.733834239828152
		
		features_values2[6]['Spikecount_std'] = 6.690285822416952/2
		features_values2[6]['mean_frequency_std'] = 10.562291538881482
		features_values2[6]['AHP_depth_abs_std'] = 9.354198154445482
		features_values2[6]['AHP_depth_abs_slow_std'] = 9.474373149612898
		features_values2[6]['AHP_slow_time_std'] = 0.1847746407036995
		features_values2[6]['AP_width_std'] = 4.8217242986858
		features_values2[6]['AP_height_std'] = 18.137501191961956
		features_values2[6]['ISI_CV_std'] = 0.3631807362672215
		features_values2[6]['inv_first_ISI_std'] = 34.032911532983164
		features_values2[6]['AP_fall_time_std'] = 1.0363378295305388
		features_values2[6]['steady_state_voltage_stimend_std'] = 9.420927820285765
		features_values2[6]['decay_time_constant_after_stim_std'] = 9.394613210941989
		
		features_values1[7]['Spikecount'] = 3.369565217391304
		features_values1[7]['mean_frequency'] = 11.458810961754287
		# features_values1[7]['AHP_depth_abs'] = -57.521739838519466
		# features_values1[7]['AHP_depth_abs_slow'] = -60.390207113306964
		# features_values1[7]['AHP_slow_time'] = 0.3333859478768642
		# features_values1[7]['AP_width'] = 2.7579438666496285
		# features_values1[7]['AP_height'] = 40.56392265613454
		# features_values1[7]['ISI_CV'] = 0.22498975418266395
		# features_values1[7]['inv_first_ISI'] = 15.476516047035451
		# features_values1[7]['AP_fall_time'] = 1.2651668519973964
		features_values1[7]['steady_state_voltage_stimend'] = -60.85488734585755
		features_values1[7]['decay_time_constant_after_stim'] = 21.31540362874316
		
		features_values2[7]['Spikecount_std'] = 4.034410121486831/2
		features_values2[7]['mean_frequency_std'] = 5.701649506179596
		# features_values2[7]['AHP_depth_abs_std'] = 9.086467319757858
		# features_values2[7]['AHP_depth_abs_slow_std'] = 8.433515069032662
		# features_values2[7]['AHP_slow_time_std'] = 0.17819762253430566
		# features_values2[7]['AP_width_std'] = 1.1722186708457936
		# features_values2[7]['AP_height_std'] = 18.232267569594782
		# features_values2[7]['ISI_CV_std'] = 0.177467811369984
		# features_values2[7]['inv_first_ISI_std'] = 21.149583420553263
		# features_values2[7]['AP_fall_time_std'] = 1.08964117551609
		features_values2[7]['steady_state_voltage_stimend_std'] = 7.3007534171312445
		features_values2[7]['decay_time_constant_after_stim_std'] = 11.470512186536931
		
		features_values1[8]['Spikecount'] = 8.021739130434783
		features_values1[8]['mean_frequency'] = 17.47538320358037
		# features_values1[8]['AHP_depth_abs'] = -55.11718163812068
		# features_values1[8]['AHP_depth_abs_slow'] = -56.94104874515491
		# features_values1[8]['AHP_slow_time'] = 0.342579703651345
		# features_values1[8]['AP_width'] = 3.097653647335267
		# features_values1[8]['AP_height'] = 38.18447403704835
		# features_values1[8]['ISI_CV'] = 0.17219712799819584
		# features_values1[8]['inv_first_ISI'] = 35.12483094526111
		# features_values1[8]['AP_fall_time'] = 1.3140592070249946
		features_values1[8]['steady_state_voltage_stimend'] = -54.003600639556545
		features_values1[8]['decay_time_constant_after_stim'] = 20.162479596388913
		
		features_values2[8]['Spikecount_std'] = 6.091651915727116/2
		features_values2[8]['mean_frequency_std'] = 8.418968352967584
		# features_values2[8]['AHP_depth_abs_std'] = 9.079669174154857
		# features_values2[8]['AHP_depth_abs_slow_std'] = 8.426339815209124
		# features_values2[8]['AHP_slow_time_std'] = 0.17935361059261942
		# features_values2[8]['AP_width_std'] = 1.697045648388051
		# features_values2[8]['AP_height_std'] = 17.381636370330813
		# features_values2[8]['ISI_CV_std'] = 0.09582519792723977
		# features_values2[8]['inv_first_ISI_std'] = 30.369455768751045
		# features_values2[8]['AP_fall_time_std'] = 1.0423072951947636
		features_values2[8]['steady_state_voltage_stimend_std'] = 8.392072647609883
		features_values2[8]['decay_time_constant_after_stim_std'] = 23.008748644471066
		
		features_values1[9]['Spikecount'] = 11.391304347826088
		features_values1[9]['mean_frequency'] = 21.551579074071693
		# features_values1[9]['AHP_depth_abs'] = -51.83975560104322
		# features_values1[9]['AHP_depth_abs_slow'] = -52.88779249387303
		# features_values1[9]['AHP_slow_time'] = 0.37530612717992234
		# features_values1[9]['AP_width'] = 4.138398264273764
		# features_values1[9]['AP_height'] = 32.55730987302576
		# features_values1[9]['ISI_CV'] = 0.26140057973597586
		# features_values1[9]['inv_first_ISI'] = 52.22021020447993
		# features_values1[9]['AP_fall_time'] = 1.1636839349052464
		features_values1[9]['steady_state_voltage_stimend'] = -49.04723567170013
		features_values1[9]['decay_time_constant_after_stim'] = 14.733834239828152
		
		features_values2[9]['Spikecount_std'] = 6.690285822416952/2
		features_values2[9]['mean_frequency_std'] = 10.562291538881482
		# features_values2[9]['AHP_depth_abs_std'] = 9.354198154445482
		# features_values2[9]['AHP_depth_abs_slow_std'] = 9.474373149612898
		# features_values2[9]['AHP_slow_time_std'] = 0.1847746407036995
		# features_values2[9]['AP_width_std'] = 4.8217242986858
		# features_values2[9]['AP_height_std'] = 18.137501191961956
		# features_values2[9]['ISI_CV_std'] = 0.3631807362672215
		# features_values2[9]['inv_first_ISI_std'] = 34.032911532983164
		# features_values2[9]['AP_fall_time_std'] = 1.0363378295305388
		features_values2[9]['steady_state_voltage_stimend_std'] = 9.420927820285765
		features_values2[9]['decay_time_constant_after_stim_std'] = 9.394613210941989
	else:
		features_values1 = efel.getMeanFeatureValues(traces_pas1, PassiveFeatures1)
		features_values2 = efel.getMeanFeatureValues(traces_pas2, PassiveFeatures2)
		features_values3 = efel.getMeanFeatureValues(traces_pas3, PassiveFeatures3)
		features_values4 = efel.getMeanFeatureValues(traces_pas4, PassiveFeatures4)
		features_values5 = efel.getMeanFeatureValues(traces_act1, ActiveFeatures1)
		features_values6 = efel.getMeanFeatureValues(traces_act2, ActiveFeatures2)
		features_values7 = efel.getMeanFeatureValues(traces_act3, ActiveFeatures3)
		
		features_values3_axon = {}
		for feat in ActiveFeatures4:
			features_values3_axon[feat] = features_values5[0][feat]
		features_values4_axon = {}
		for feat in ActiveFeatures5:
			features_values4_axon[feat] = features_values6[0][feat]
		features_values5_axon = {}
		for feat in ActiveFeatures6:
			features_values5_axon[feat] = features_values7[0][feat]
		
		features_values1.extend(features_values2)
		features_values1.extend(features_values3)
		features_values1.extend(features_values4)
		features_values1.extend(features_values5)
		features_values1.extend(features_values6)
		features_values1.extend(features_values7)
		features_values1.extend([features_values3_axon])
		features_values1.extend([features_values4_axon])
		features_values1.extend([features_values5_axon])
		features_values2=0 # i.e. since no stdev values for this option
	
	return features_values1, features_values2

active_features, active_features_stds = get_active_features(data_active_dict,1)

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
pasapical_mech = ephys.mechanisms.NrnMODMechanism(
	name ='pasapical',
	suffix ='pas',
	locations = [apical_loc]
)
pasaxonal_mech = ephys.mechanisms.NrnMODMechanism(
	name ='pasaxonal',
	suffix ='pas',
	locations = [axonal_loc]
)
act_mechs['passomatic_mech'] = passomatic_mech
act_mechs['pasbasal_mech'] = pasbasal_mech
act_mechs['pasapical_mech'] = pasapical_mech
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
	value = 0.9,
	frozen = True
)
cm_basal_param = ephys.parameters.NrnSectionParameter(
	name ='cmbasal',
	param_name ='cm',
	locations = [basal_loc],
	value = 0.9,
	frozen = True
)
cm_apical_param = ephys.parameters.NrnSectionParameter(
	name ='cmapical',
	param_name ='cm',
	locations = [apical_loc],
	value = 0.9,
	frozen = True
)
cm_axonal_param = ephys.parameters.NrnSectionParameter(
	name ='cmaxonal',
	param_name ='cm',
	locations = [axonal_loc],
	value = 0.9,
	frozen = True
)
g_pas_param = ephys.parameters.NrnSectionParameter(
	name ='g_pas',
	param_name ='g_pas',
	locations = [all_loc],
	value = 0.00015,
	bounds=[0.000001, 0.0002],
	frozen = False
)
e_pas_param = ephys.parameters.NrnSectionParameter(
	name ='e_pas',
	param_name ='e_pas',
	locations = [all_loc],
	value = -73,
	bounds=[-100, -72],
	frozen = False
)

######### Set Up Active Parameters to Optimize #########
scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(name='Ihscaler',distribution="(0.5 + 24/(1 + math.exp(({distance} - 950)/-285))) * {value}") # Absolute Sigmoidal

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
				bounds=[0, 1],
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
			act_params["{0}{1}_param".format('slopeh_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='slopeh_NaTg'+x['section'],
				param_name ='slopeh_NaTg',
				locations = [loc],
				value = 6,
				frozen = True)
		elif x['section'] == 'axonal':
			act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
				name =x['name']+x['section'],
				param_name =x['name'],
				locations = [loc],
				value = 0.15,
				bounds=[0, 1],
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
			act_params["{0}{1}_param".format('slopeh_NaTg',x['section'])]=ephys.parameters.NrnSectionParameter(
				name ='slopeh_NaTg'+x['section'],
				param_name ='slopeh_NaTg',
				locations = [loc],
				value = 6,
				frozen = True)
	elif x['name'] == 'gbar_Nap':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-3,
			bounds=[0, 1e-2],
			frozen = False)
	elif x['name'] == 'gbar_K_P':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.1,
			bounds=[0, 1.5],
			frozen = False)
	elif x['name'] == 'gbar_K_T':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-2,
			bounds=[0, 1.5],
			frozen = False)
	elif x['name'] == 'gbar_SK':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-4,
			bounds=[0, 1.5],
			frozen = False)
	elif x['name'] == 'gbar_Kv3_1':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 0.6,
			bounds=[0, 1.5],
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
			value = 1e-5,
			bounds=[0, 1e-4],
			frozen = False)
	elif x['name'] == 'gbar_Ca_LVA':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-4,
			bounds=[0, 1e-2],
			frozen = False)
	elif x['name'] == 'gbar_Im':
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnSectionParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			locations = [loc],
			value = 1e-5,
			bounds=[0, 5e-4],
			frozen = False)
	elif (x['name'] == 'gbar_Ih') and (x['section'] == 'all'):
		act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnRangeParameter(
			name =x['name']+x['section'],
			param_name =x['name'],
			value_scaler=scaler,
			locations = [loc],
			value = 0,
			bounds=[0, 0.0001],
			frozen = False)
		act_params["{0}{1}_param".format('shift1_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift1_Ih'+x['section'],
			param_name ='shift1_Ih',
			locations = [loc],
			value = 144.76545935424588,
			frozen = True)
		act_params["{0}{1}_param".format('shift2_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift2_Ih'+x['section'],
			param_name ='shift2_Ih',
			locations = [loc],
			value = 14.382865335237211,
			frozen = True)
		act_params["{0}{1}_param".format('shift3_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift3_Ih'+x['section'],
			param_name ='shift3_Ih',
			locations = [loc],
			value = -28.179477866349245,
			frozen = True)
		act_params["{0}{1}_param".format('shift4_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift4_Ih'+x['section'],
			param_name ='shift4_Ih',
			locations = [loc],
			value = 99.18311385307702,
			frozen = True)
		act_params["{0}{1}_param".format('shift5_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift5_Ih'+x['section'],
			param_name ='shift5_Ih',
			locations = [loc],
			value = 16.42000098505615,
			frozen = True)
		act_params["{0}{1}_param".format('shift6_Ih',x['section'])]=ephys.parameters.NrnSectionParameter(
			name ='shift6_Ih'+x['section'],
			param_name ='shift6_Ih',
			locations = [loc],
			value = 26.699880497099517,
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

mod_params = [celsius_param, v_init_param, ena_paramsomatic, ena_paramaxonal, ek_paramsomatic, ek_paramaxonal, e_pas_param, Ra_param, cm_somatic_param, cm_basal_param, cm_apical_param, cm_axonal_param, g_pas_param]
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
for protocol_name, amplitude in [('step1', active_steps[0]), ('step2', active_steps[1]), ('step3', active_steps[2]), ('step4', active_steps[3]), ('step5', active_steps[4]), ('step6', active_steps[5]), ('step7', active_steps[6])]:
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

active_sevenstep_protocol = ephys.protocols.SequenceProtocol(name='sevenstep', protocols=sweep_protocols_act)

#################################
### Set-up Fitness Calculator ###
#################################

def define_fitness_calculator(sweeps,feature_definitions,feature_stds):
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
			std = feature_stds[c][efel_feature_name+'_std']
			
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
				threshold=threshold,
				stimulus_current=stim_amp)
			
			fname = {}
			fname[efel_feature_name] = feature
			if stim_amp < 0:
				features_pas.append(fname)
			elif stim_amp > 0:
				features_act.append(fname)
		
		c = c + 1
	
	for sweep in sweeps[4:]:
		stim_start = sweep.stimuli[0].step_delay
		stim_end = stim_start + sweep.stimuli[0].step_duration
		stim_amp = sweep.stimuli[0].step_amplitude
		for efel_feature_name in feature_definitions[c]:
			feature_name = '%s.%s' % (sweep.name, efel_feature_name)
			
			mean = feature_definitions[c][efel_feature_name]
			std = feature_stds[c][efel_feature_name+'_std']
			
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
				threshold=threshold,
				stimulus_current=stim_amp)
			
			fname = {}
			fname[efel_feature_name] = feature
			features_act.append(fname)
		
		c = c + 1
	
	# Passive objectives
	for efel_feature_name in feature_definitions[0]:
		tempfeatlist = [d[efel_feature_name] for d in features_pas if efel_feature_name in d]
		objective = ephys.objectives.MaxObjective(
		efel_feature_name,
		tempfeatlist)
		objectives.append(objective)
	
	# Active objectives
	for efel_feature_name in feature_definitions[4]:
		tempfeatlist = [d[efel_feature_name] for d in features_act if efel_feature_name in d]
		objective = ephys.objectives.MaxObjective(
		efel_feature_name,
		tempfeatlist)
		objectives.append(objective)
	
	fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
	
	return fitcalc, stdvec

fitness_calculator_act, stdvec_act = define_fitness_calculator(sweep_protocols_act,active_features,active_features_stds)

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
		fitness_protocols={active_sevenstep_protocol.name: active_sevenstep_protocol},
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
		seed=9189)

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
	plt.subplot(4,1,1)
	plt.plot(responses['step1.soma.v']['time'], responses['step1.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-125,-25)
	plt.subplot(4,1,2)
	plt.plot(responses['step2.soma.v']['time'], responses['step2.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-125,-25)
	plt.subplot(4,1,3)
	plt.plot(responses['step3.soma.v']['time'], responses['step3.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-125,-25)
	plt.subplot(4,1,4)
	plt.plot(responses['step4.soma.v']['time'], responses['step4.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.ylim(-125,-25)
	plt.tight_layout()

def plot_responses_act(responses):
	plt.subplot(3,1,1)
	plt.plot(responses['step5.soma.v']['time'], responses['step5.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.subplot(3,1,2)
	plt.plot(responses['step6.soma.v']['time'], responses['step6.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.subplot(3,1,3)
	plt.plot(responses['step7.soma.v']['time'], responses['step7.soma.v']['voltage'], color='r')
	plt.xlim(stimstart-200,stimend+200)
	plt.tight_layout()

def plot_responses_act2(responses):
	plt.subplot(3,1,1)
	plt.plot(responses['step5.axon.v']['time'], responses['step5.axon.v']['voltage'], color='m')
	plt.plot(responses['step5.soma.v']['time'], responses['step5.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.subplot(3,1,2)
	plt.plot(responses['step6.axon.v']['time'], responses['step6.axon.v']['voltage'], color='m')
	plt.plot(responses['step6.soma.v']['time'], responses['step6.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.subplot(3,1,3)
	plt.plot(responses['step7.axon.v']['time'], responses['step7.axon.v']['voltage'], color='m')
	plt.plot(responses['step7.soma.v']['time'], responses['step7.soma.v']['voltage'], color='r')
	plt.xlim(stimstart,stimstart+200)
	plt.tight_layout()

active_steps = [-0.4,-0.3,-0.2,-0.05,0.1,0.2,0.3,0.1,0.2,0.3] # nA
def get_active_features2(data):
	traces_pas1 = []
	traces_pas2 = []
	traces_pas3 = []
	traces_pas4 = []
	traces_act1 = []
	traces_act2 = []
	traces_act3 = []
	traces_act4 = []
	traces_act5 = []
	traces_act6 = []
	for step_name, step_traces in data.items():
		trace = {}
		trace['T'] = data[step_name]['T']
		trace['V'] = data[step_name]['V']
		trace['stim_start'] = [stimstart]
		trace['stim_end'] = [stimend]
		trace['name'] = [step_name]
		trace['stimulus_current'] = [active_steps[step_name]]
		if step_name == 0:
			traces_pas1.append(trace)
		elif step_name == 1:
			traces_pas2.append(trace)
		elif step_name == 2:
			traces_pas3.append(trace)
		elif step_name == 3:
			traces_pas4.append(trace)
		elif step_name == 4:
			traces_act1.append(trace)
		elif step_name == 5:
			traces_act2.append(trace)
		elif step_name == 6:
			traces_act3.append(trace)
		elif step_name == 7:
			traces_act4.append(trace)
		elif step_name == 8:
			traces_act5.append(trace)
		elif step_name == 9:
			traces_act6.append(trace)
	
	features_values_pas1 = efel.getMeanFeatureValues(traces_pas1, PassiveFeatures1)
	features_values_pas2 = efel.getMeanFeatureValues(traces_pas2, PassiveFeatures2)
	features_values_pas3 = efel.getMeanFeatureValues(traces_pas3, PassiveFeatures3)
	features_values_pas4 = efel.getMeanFeatureValues(traces_pas4, PassiveFeatures4)
	features_values_act1 = efel.getMeanFeatureValues(traces_act1, ActiveFeatures1)
	features_values_act2 = efel.getMeanFeatureValues(traces_act2, ActiveFeatures2)
	features_values_act3 = efel.getMeanFeatureValues(traces_act3, ActiveFeatures3)
	features_values_act4 = efel.getMeanFeatureValues(traces_act4, ActiveFeatures4)
	features_values_act5 = efel.getMeanFeatureValues(traces_act5, ActiveFeatures5)
	features_values_act6 = efel.getMeanFeatureValues(traces_act6, ActiveFeatures6)
	
	features_values_pas1.extend(features_values_pas2)
	features_values_pas1.extend(features_values_pas3)
	features_values_pas1.extend(features_values_pas4)
	features_values_pas1.extend(features_values_act1)
	features_values_pas1.extend(features_values_act2)
	features_values_pas1.extend(features_values_act3)
	features_values_pas1.extend(features_values_act4)
	features_values_pas1.extend(features_values_act5)
	features_values_pas1.extend(features_values_act6)
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

responses = active_sevenstep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict, sim=nrn)
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
sweep_data6['V'] = responses['step6.soma.v']['voltage']
sweep_data6['T'] = responses['step6.soma.v']['time']
sweep_data7 = {}
sweep_data7['V'] = responses['step7.soma.v']['voltage']
sweep_data7['T'] = responses['step7.soma.v']['time']
sweep_data8 = {}
sweep_data8['V'] = responses['step5.axon.v']['voltage']
sweep_data8['T'] = responses['step5.axon.v']['time']
sweep_data9 = {}
sweep_data9['V'] = responses['step6.axon.v']['voltage']
sweep_data9['T'] = responses['step6.axon.v']['time']
sweep_data10 = {}
sweep_data10['V'] = responses['step7.axon.v']['voltage']
sweep_data10['T'] = responses['step7.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
data[6] = sweep_data7
data[7] = sweep_data8
data[8] = sweep_data9
data[9] = sweep_data10
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

responses = active_sevenstep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict2, sim=nrn)
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
sweep_data6['V'] = responses['step6.soma.v']['voltage']
sweep_data6['T'] = responses['step6.soma.v']['time']
sweep_data7 = {}
sweep_data7['V'] = responses['step7.soma.v']['voltage']
sweep_data7['T'] = responses['step7.soma.v']['time']
sweep_data8 = {}
sweep_data8['V'] = responses['step5.axon.v']['voltage']
sweep_data8['T'] = responses['step5.axon.v']['time']
sweep_data9 = {}
sweep_data9['V'] = responses['step6.axon.v']['voltage']
sweep_data9['T'] = responses['step6.axon.v']['time']
sweep_data10 = {}
sweep_data10['V'] = responses['step7.axon.v']['voltage']
sweep_data10['T'] = responses['step7.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
data[6] = sweep_data7
data[7] = sweep_data8
data[8] = sweep_data9
data[9] = sweep_data10
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

responses = active_sevenstep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict3, sim=nrn)
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
sweep_data6['V'] = responses['step6.soma.v']['voltage']
sweep_data6['T'] = responses['step6.soma.v']['time']
sweep_data7 = {}
sweep_data7['V'] = responses['step7.soma.v']['voltage']
sweep_data7['T'] = responses['step7.soma.v']['time']
sweep_data8 = {}
sweep_data8['V'] = responses['step5.axon.v']['voltage']
sweep_data8['T'] = responses['step5.axon.v']['time']
sweep_data9 = {}
sweep_data9['V'] = responses['step6.axon.v']['voltage']
sweep_data9['T'] = responses['step6.axon.v']['time']
sweep_data10 = {}
sweep_data10['V'] = responses['step7.axon.v']['voltage']
sweep_data10['T'] = responses['step7.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
data[6] = sweep_data7
data[7] = sweep_data8
data[8] = sweep_data9
data[9] = sweep_data10
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

responses = active_sevenstep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict4, sim=nrn)
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
sweep_data6['V'] = responses['step6.soma.v']['voltage']
sweep_data6['T'] = responses['step6.soma.v']['time']
sweep_data7 = {}
sweep_data7['V'] = responses['step7.soma.v']['voltage']
sweep_data7['T'] = responses['step7.soma.v']['time']
sweep_data8 = {}
sweep_data8['V'] = responses['step5.axon.v']['voltage']
sweep_data8['T'] = responses['step5.axon.v']['time']
sweep_data9 = {}
sweep_data9['V'] = responses['step6.axon.v']['voltage']
sweep_data9['T'] = responses['step6.axon.v']['time']
sweep_data10 = {}
sweep_data10['V'] = responses['step7.axon.v']['voltage']
sweep_data10['T'] = responses['step7.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
data[6] = sweep_data7
data[7] = sweep_data8
data[8] = sweep_data9
data[9] = sweep_data10
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

responses = active_sevenstep_protocol.run(cell_model = Cell_Model, param_values=best_ind_dict5, sim=nrn)
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
sweep_data6['V'] = responses['step6.soma.v']['voltage']
sweep_data6['T'] = responses['step6.soma.v']['time']
sweep_data7 = {}
sweep_data7['V'] = responses['step7.soma.v']['voltage']
sweep_data7['T'] = responses['step7.soma.v']['time']
sweep_data8 = {}
sweep_data8['V'] = responses['step5.axon.v']['voltage']
sweep_data8['T'] = responses['step5.axon.v']['time']
sweep_data9 = {}
sweep_data9['V'] = responses['step6.axon.v']['voltage']
sweep_data9['T'] = responses['step6.axon.v']['time']
sweep_data10 = {}
sweep_data10['V'] = responses['step7.axon.v']['voltage']
sweep_data10['T'] = responses['step7.axon.v']['time']
data[0] = sweep_data1
data[1] = sweep_data2
data[2] = sweep_data3
data[3] = sweep_data4
data[4] = sweep_data5
data[5] = sweep_data6
data[6] = sweep_data7
data[7] = sweep_data8
data[8] = sweep_data9
data[9] = sweep_data10
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
