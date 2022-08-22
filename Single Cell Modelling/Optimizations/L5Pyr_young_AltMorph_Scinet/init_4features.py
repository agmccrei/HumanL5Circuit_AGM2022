###################################
### Setup Optimization Features ###
###################################

# Specify optimization method
# if fitting to single cell data from ABI, enter target_feature_type = 'Automatic' (note: pkl files are required for this option, but features are calculated automatically)
# if fitting to population data, or just entering features manually, enter target_feature_type = 'Manual' (note: feature mean/sd values are inputted manually below)
target_feature_type = 'Manual'

# Specify stimulus start/end times
stimstart = 1000
stimend = 1600

# Enter somatic current steps and steps where features from non-somatic compartments are also evaluated
c_soma = [-0.4,-0.3,-0.2,-0.05,0.1,0.2,0.3] # pyramidal steps
c_axon = [0.1,0.2,0.3] # entries must match with corresponding somatic steps
active_steps = c_soma + c_axon  # Concatenate c_soma and c_axon lists

# create recording location specifiers
rec1name = 'soma'
rec2name = 'axon'
c_somalocs = [rec1name for _ in range(0,len(c_soma))]
c_axonlocs = [rec2name for _ in range(0,len(c_axon))]
RecLocs = c_somalocs + c_axonlocs

# create step number specifiers (i.e. [1,2,3,...])
s_soma = [i for i in range(1,len(c_soma)+1)]
s_axon = []
for i,cs in enumerate(c_soma):
	for ca in c_axon:
		if cs == ca:
			s_axon.append(s_soma[i])

StepNums = s_soma + s_axon # Corresponding step numbers for each recording

# Load data if using single cell data
if target_feature_type == 'Automatic':
	data_passive_dict = pickle.load(open("experimental_data/passive.pkl","rb"))
	data_active_dict = pickle.load(open("experimental_data/active.pkl","rb"))
	
	# Create new dict of all somatic experimental traces (i.e. merge passive and active)
	d_a_d = {}
	for i in range(0,len(data_active_dict)):
		d_a_d[i+len(data_passive_dict)] = data_active_dict[i]
	data = {**data_passive_dict, **d_a_d}
	
	# Duplicate in somatic trace(s) for axon
	step_name = [s-1 for s in s_axon]
	for s in step_name:
		d_a_d_axon = {}
		d_a_d_axon[len(data)] = data[s]
		data = {**data, **d_a_d_axon}

# Feature list builder function
def get_features():
	if target_feature_type == 'Automatic':
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
		ActiveFeatures4 = ['Spikecount', 'AHP_depth_abs', 'AP_width', 'AP_height']
		
		AllFeatNames = [PassiveFeatures1,PassiveFeatures2,ActiveFeatures1,ActiveFeatures2,ActiveFeatures3,ActiveFeatures4]
		
		features_values_mean = []
		for step_name, step_traces in data.items():
			trace = {}
			trace['T'] = data[step_name]['T']
			trace['V'] = data[step_name]['V']
			trace['stim_start'] = [stimstart]
			trace['stim_end'] = [stimend]
			trace['name'] = [step_name]
			trace['stimulus_current'] = [active_steps[step_name]]
			feature_values = efel.getMeanFeatureValues([trace], AllFeatNames[step_name])
			features_values_mean.extend(feature_values)
		
		# Construct standard deviation values to use for weighting features
		features_values_sd = numpy.array([dict([]) for _ in range(0,len(active_steps))])
		
		for i,featset in enumerate(AllFeatNames):
			for featname in featset:
				if featname == 'ohmic_input_resistance_vb_ssse':
					sd = 1
				elif featname == 'sag_amplitude':
					sd = 0.01
				elif featname == 'voltage_base':
					sd = 0.01
				elif featname == 'decay_time_constant_after_stim':
					sd = 0.1
				elif featname == 'voltage_deflection':
					sd = 0.1
				elif featname == 'steady_state_hyper':
					sd = 0.1
				elif featname == 'Spikecount':
					sd = 0.01
				elif featname == 'AHP_depth_abs':
					sd = 1
				elif featname == 'AP_width':
					sd = 0.01
				elif featname == 'AP_height':
					sd = 1
				elif featname == 'ISI_CV':
					sd = 0.01
				elif featname == 'inv_first_ISI':
					sd = 1
				elif featname == 'inv_second_ISI':
					sd = 1
				elif featname == 'AHP_depth_abs_slow':
					sd = 0.1
				elif featname == 'AHP_slow_time':
					sd = 0.05
				elif featname == 'AHP_time_from_peak':
					sd = 0.03
				elif featname == 'adaptation_index':
					sd = 0.001
				else:
					sd = abs(mean*0.1)
				features_values_sd[i][featname] = sd
		
	elif target_feature_type == 'Manual':
		features_values_mean = numpy.array([dict([]) for _ in range(0,len(active_steps))])
		features_values_sd = numpy.array([dict([]) for _ in range(0,len(active_steps))])
		
		# somatic passive
		features_values_mean[0]['sag_amplitude'] = 3.6775989649854615
		features_values_mean[0]['sag_ratio1'] = 0.12221060716408808
		features_values_mean[0]['ohmic_input_resistance_vb_ssse'] = 72.28615724524
		features_values_mean[0]['voltage_base'] = -71.87957306165225
		
		features_values_sd[0]['sag_amplitude'] = 1.8254970526175351/10
		features_values_sd[0]['sag_ratio1'] = 0.05010451751355496
		features_values_sd[0]['ohmic_input_resistance_vb_ssse'] = 33.454297498857684
		features_values_sd[0]['voltage_base'] = 5.007265435844746/2
		
		features_values_mean[1]['sag_amplitude'] = 2.927076006742862
		features_values_mean[1]['sag_ratio1'] = 0.11930780081786073
		features_values_mean[1]['ohmic_input_resistance_vb_ssse'] = 79.08759551159801
		features_values_mean[1]['voltage_base'] = -72.1908270945765
		
		features_values_sd[1]['sag_amplitude'] = 1.5823616224205286/10
		features_values_sd[1]['sag_ratio1'] = 0.05582615959936093
		features_values_sd[1]['ohmic_input_resistance_vb_ssse'] = 35.82516245212583
		features_values_sd[1]['voltage_base'] = 5.147306955246066/2
		
		features_values_mean[2]['sag_amplitude'] = 2.1625007279520525
		features_values_mean[2]['sag_ratio1'] = 0.11856984168498061
		features_values_mean[2]['ohmic_input_resistance_vb_ssse'] = 88.07095848954897
		features_values_mean[2]['voltage_base'] = -72.23412015231611
		
		features_values_sd[2]['sag_amplitude'] = 1.2417425256840757/10
		features_values_sd[2]['sag_ratio1'] = 0.056727669541529774
		features_values_sd[2]['ohmic_input_resistance_vb_ssse'] = 41.148312874114666
		features_values_sd[2]['voltage_base'] = 5.106214743866047/2
		
		features_values_mean[3]['sag_amplitude'] = 0.9321999814343563
		features_values_mean[3]['sag_ratio1'] = 0.16311846985029396
		features_values_mean[3]['ohmic_input_resistance_vb_ssse'] = 113.62408571001566
		features_values_mean[3]['voltage_base'] = -72.52907339557296
		
		features_values_sd[3]['sag_amplitude'] = 0.5493372552313776/5
		features_values_sd[3]['sag_ratio1'] = 0.08759962622859466
		features_values_sd[3]['ohmic_input_resistance_vb_ssse'] = 64.55495760776186
		features_values_sd[3]['voltage_base'] = 5.106458376803106/2
		
		# somatic active
		features_values_mean[4]['Spikecount'] = 3.369565217391304
		features_values_mean[4]['mean_frequency'] = 11.458810961754287
		features_values_mean[4]['AHP_depth_abs'] = -57.521739838519466
		features_values_mean[4]['AHP_depth_abs_slow'] = -60.390207113306964
		features_values_mean[4]['AHP_slow_time'] = 0.3333859478768642
		features_values_mean[4]['AP_width'] = 2.7579438666496285
		features_values_mean[4]['AP_height'] = 40.56392265613454
		features_values_mean[4]['ISI_CV'] = 0.22498975418266395
		features_values_mean[4]['inv_first_ISI'] = 15.476516047035451
		features_values_mean[4]['AP_fall_time'] = 1.2651668519973964
		features_values_mean[4]['steady_state_voltage_stimend'] = -60.85488734585755
		features_values_mean[4]['decay_time_constant_after_stim'] = 21.31540362874316
		
		features_values_sd[4]['Spikecount'] = 4.034410121486831
		features_values_sd[4]['mean_frequency'] = 5.701649506179596
		features_values_sd[4]['AHP_depth_abs'] = 9.086467319757858
		features_values_sd[4]['AHP_depth_abs_slow'] = 8.433515069032662
		features_values_sd[4]['AHP_slow_time'] = 0.17819762253430566
		features_values_sd[4]['AP_width'] = 1.1722186708457936
		features_values_sd[4]['AP_height'] = 18.232267569594782
		features_values_sd[4]['ISI_CV'] = 0.177467811369984
		features_values_sd[4]['inv_first_ISI'] = 21.149583420553263
		features_values_sd[4]['AP_fall_time'] = 1.08964117551609
		features_values_sd[4]['steady_state_voltage_stimend'] = 7.3007534171312445
		features_values_sd[4]['decay_time_constant_after_stim'] = 11.470512186536931
		
		features_values_mean[5]['Spikecount'] = 8.021739130434783
		features_values_mean[5]['mean_frequency'] = 17.47538320358037
		features_values_mean[5]['AHP_depth_abs'] = -55.11718163812068
		features_values_mean[5]['AHP_depth_abs_slow'] = -56.94104874515491
		features_values_mean[5]['AHP_slow_time'] = 0.342579703651345
		features_values_mean[5]['AP_width'] = 3.097653647335267
		features_values_mean[5]['AP_height'] = 38.18447403704835
		features_values_mean[5]['ISI_CV'] = 0.17219712799819584
		features_values_mean[5]['inv_first_ISI'] = 35.12483094526111
		features_values_mean[5]['AP_fall_time'] = 1.3140592070249946
		features_values_mean[5]['steady_state_voltage_stimend'] = -54.003600639556545
		features_values_mean[5]['decay_time_constant_after_stim'] = 20.162479596388913
		
		features_values_sd[5]['Spikecount'] = 6.091651915727116/2
		features_values_sd[5]['mean_frequency'] = 8.418968352967584
		features_values_sd[5]['AHP_depth_abs'] = 9.079669174154857
		features_values_sd[5]['AHP_depth_abs_slow'] = 8.426339815209124
		features_values_sd[5]['AHP_slow_time'] = 0.17935361059261942
		features_values_sd[5]['AP_width'] = 1.697045648388051
		features_values_sd[5]['AP_height'] = 17.381636370330813
		features_values_sd[5]['ISI_CV'] = 0.09582519792723977
		features_values_sd[5]['inv_first_ISI'] = 30.369455768751045
		features_values_sd[5]['AP_fall_time'] = 1.0423072951947636
		features_values_sd[5]['steady_state_voltage_stimend'] = 8.392072647609883
		features_values_sd[5]['decay_time_constant_after_stim'] = 23.008748644471066
		
		features_values_mean[6]['Spikecount'] = 11.391304347826088
		features_values_mean[6]['mean_frequency'] = 21.551579074071693
		features_values_mean[6]['AHP_depth_abs'] = -51.83975560104322
		features_values_mean[6]['AHP_depth_abs_slow'] = -52.88779249387303
		features_values_mean[6]['AHP_slow_time'] = 0.37530612717992234
		features_values_mean[6]['AP_width'] = 4.138398264273764
		features_values_mean[6]['AP_height'] = 32.55730987302576
		features_values_mean[6]['ISI_CV'] = 0.26140057973597586
		features_values_mean[6]['inv_first_ISI'] = 52.22021020447993
		features_values_mean[6]['AP_fall_time'] = 1.1636839349052464
		features_values_mean[6]['steady_state_voltage_stimend'] = -49.04723567170013
		features_values_mean[6]['decay_time_constant_after_stim'] = 14.733834239828152
		
		features_values_sd[6]['Spikecount'] = 6.690285822416952/2
		features_values_sd[6]['mean_frequency'] = 10.562291538881482
		features_values_sd[6]['AHP_depth_abs'] = 9.354198154445482
		features_values_sd[6]['AHP_depth_abs_slow'] = 9.474373149612898
		features_values_sd[6]['AHP_slow_time'] = 0.1847746407036995
		features_values_sd[6]['AP_width'] = 4.8217242986858
		features_values_sd[6]['AP_height'] = 18.137501191961956
		features_values_sd[6]['ISI_CV'] = 0.3631807362672215
		features_values_sd[6]['inv_first_ISI'] = 34.032911532983164
		features_values_sd[6]['AP_fall_time'] = 1.0363378295305388
		features_values_sd[6]['steady_state_voltage_stimend'] = 9.420927820285765
		features_values_sd[6]['decay_time_constant_after_stim'] = 9.394613210941989
		
		# axon features
		features_values_mean[7]['Spikecount'] = 3.369565217391304
		features_values_mean[7]['mean_frequency'] = 11.458810961754287
		features_values_mean[7]['steady_state_voltage_stimend'] = -60.85488734585755
		features_values_mean[7]['decay_time_constant_after_stim'] = 21.31540362874316
		
		features_values_sd[7]['Spikecount'] = 4.034410121486831/2
		features_values_sd[7]['mean_frequency'] = 5.701649506179596
		features_values_sd[7]['steady_state_voltage_stimend'] = 7.3007534171312445
		features_values_sd[7]['decay_time_constant_after_stim'] = 11.470512186536931
		
		features_values_mean[8]['Spikecount'] = 8.021739130434783
		features_values_mean[8]['mean_frequency'] = 17.47538320358037
		features_values_mean[8]['steady_state_voltage_stimend'] = -54.003600639556545
		features_values_mean[8]['decay_time_constant_after_stim'] = 20.162479596388913
		
		features_values_sd[8]['Spikecount'] = 6.091651915727116/2
		features_values_sd[8]['mean_frequency'] = 8.418968352967584
		features_values_sd[8]['steady_state_voltage_stimend'] = 8.392072647609883
		features_values_sd[8]['decay_time_constant_after_stim'] = 23.008748644471066
		
		features_values_mean[9]['Spikecount'] = 11.391304347826088
		features_values_mean[9]['mean_frequency'] = 21.551579074071693
		features_values_mean[9]['steady_state_voltage_stimend'] = -49.04723567170013
		features_values_mean[9]['decay_time_constant_after_stim'] = 14.733834239828152
		
		features_values_sd[9]['Spikecount'] = 6.690285822416952/2
		features_values_sd[9]['mean_frequency'] = 10.562291538881482
		features_values_sd[9]['steady_state_voltage_stimend'] = 9.420927820285765
		features_values_sd[9]['decay_time_constant_after_stim'] = 9.394613210941989
		
		# Construct list of all feature names
		AllFeatNames = []
		for l in features_values_mean:
			AllFeatNames.append(list(l.keys()))
		
	return list(features_values_mean), list(features_values_sd), list(AllFeatNames)

all_features, all_features_sd, AllFeatNames = get_features()
