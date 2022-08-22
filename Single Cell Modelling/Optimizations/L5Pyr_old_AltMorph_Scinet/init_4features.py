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
		features_values_mean[0]['sag_amplitude'] = 5.278100726466888
		features_values_mean[0]['sag_ratio1'] = 0.1481642228342842
		features_values_mean[0]['ohmic_input_resistance_vb_ssse'] = 74.03527452374264
		features_values_mean[0]['voltage_base'] = -70.88430730663849
		
		features_values_sd[0]['sag_amplitude'] = 2.6660693952549757/10
		features_values_sd[0]['sag_ratio1'] = 0.04456314652499393
		features_values_sd[0]['ohmic_input_resistance_vb_ssse'] = 24.36723464645282
		features_values_sd[0]['voltage_base'] = 4.847965941570795/2
		
		features_values_mean[1]['sag_amplitude'] = 4.14297392783007
		features_values_mean[1]['sag_ratio1'] = 0.1430363798705737
		features_values_mean[1]['ohmic_input_resistance_vb_ssse'] = 80.30606348785402
		features_values_mean[1]['voltage_base'] = -70.77824081894033
		
		features_values_sd[1]['sag_amplitude'] = 2.4048490063182313/10
		features_values_sd[1]['sag_ratio1'] = 0.05418255221975284
		features_values_sd[1]['ohmic_input_resistance_vb_ssse'] = 26.22398634726206
		features_values_sd[1]['voltage_base'] = 5.050300427120481/2
		
		features_values_mean[2]['sag_amplitude'] = 3.0893530826949833
		features_values_mean[2]['sag_ratio1'] = 0.14641767702365857
		features_values_mean[2]['ohmic_input_resistance_vb_ssse'] = 89.28021086034806
		features_values_mean[2]['voltage_base'] = -70.75919314447394
		
		features_values_sd[2]['sag_amplitude'] = 1.7988838813077013/10
		features_values_sd[2]['sag_ratio1'] = 0.058254740064004454
		features_values_sd[2]['ohmic_input_resistance_vb_ssse'] = 30.009528125222737
		features_values_sd[2]['voltage_base'] = 5.234119677048471/2
		
		features_values_mean[3]['sag_amplitude'] = 1.5375874896612003
		features_values_mean[3]['sag_ratio1'] = 0.22016542016166685
		features_values_mean[3]['ohmic_input_resistance_vb_ssse'] = 110.4504308246921
		features_values_mean[3]['voltage_base'] = -71.03105667954445
		
		features_values_sd[3]['sag_amplitude'] = 0.850533526773676/5
		features_values_sd[3]['sag_ratio1'] = 0.09671918252213504
		features_values_sd[3]['ohmic_input_resistance_vb_ssse'] = 46.54981874949187
		features_values_sd[3]['voltage_base'] = 4.800949900675258/2
		
		# somatic active
		features_values_mean[4]['Spikecount'] = 3.727272727272727
		features_values_mean[4]['mean_frequency'] = 9.602766320585369
		features_values_mean[4]['AHP_depth_abs'] = -58.07434994377865
		features_values_mean[4]['AHP_depth_abs_slow'] = -60.39359192619093
		features_values_mean[4]['AHP_slow_time'] = 0.2649655033059525
		features_values_mean[4]['AP_width'] = 2.4566666666670987
		features_values_mean[4]['AP_height'] = 44.47247679980026
		features_values_mean[4]['ISI_CV'] = 0.17641360052825844
		features_values_mean[4]['inv_first_ISI'] = 11.19333032135457
		features_values_mean[4]['AP_fall_time'] = 2.008958333333663
		features_values_mean[4]['steady_state_voltage_stimend'] = -58.40119724054735
		features_values_mean[4]['decay_time_constant_after_stim'] = 17.426882279617683
		
		features_values_sd[4]['Spikecount'] = 2.7990553306073913
		features_values_sd[4]['mean_frequency'] = 3.194504470538059
		features_values_sd[4]['AHP_depth_abs'] = 6.353539415690842
		features_values_sd[4]['AHP_depth_abs_slow'] = 5.809293847391061
		features_values_sd[4]['AHP_slow_time'] = 0.09423464665336653
		features_values_sd[4]['AP_width'] = 0.8745121655975552
		features_values_sd[4]['AP_height'] = 7.394850554959106
		features_values_sd[4]['ISI_CV'] = 0.13409875646853456
		features_values_sd[4]['inv_first_ISI'] = 11.588141398703439
		features_values_sd[4]['AP_fall_time'] = 1.1452495293059304
		features_values_sd[4]['steady_state_voltage_stimend'] = 8.590194216930989
		features_values_sd[4]['decay_time_constant_after_stim'] = 11.993943660719102
		
		features_values_mean[5]['Spikecount'] = 7.454545454545454
		features_values_mean[5]['mean_frequency'] = 18.2489821761524
		features_values_mean[5]['AHP_depth_abs'] = -55.13927279609413
		features_values_mean[5]['AHP_depth_abs_slow'] = -56.73911699122267
		features_values_mean[5]['AHP_slow_time'] = 0.3007859939045979
		features_values_mean[5]['AP_width'] = 2.823222222222639
		features_values_mean[5]['AP_height'] = 38.981035646708285
		features_values_mean[5]['ISI_CV'] = 0.15304857074487668
		features_values_mean[5]['inv_first_ISI'] = 32.339931952506134
		features_values_mean[5]['AP_fall_time'] = 2.0331047008549485
		features_values_mean[5]['steady_state_voltage_stimend'] = -53.353307066898694
		features_values_mean[5]['decay_time_constant_after_stim'] = 14.601388658491764
		
		features_values_sd[5]['Spikecount'] = 4.163993632945013/2
		features_values_sd[5]['mean_frequency'] = 3.8945348914456726
		features_values_sd[5]['AHP_depth_abs'] = 8.088493834520389
		features_values_sd[5]['AHP_depth_abs_slow'] = 8.089093654518868
		features_values_sd[5]['AHP_slow_time'] = 0.10255675046087279
		features_values_sd[5]['AP_width'] = 1.4785934632719298
		features_values_sd[5]['AP_height'] = 9.283539200158264
		features_values_sd[5]['ISI_CV'] = 0.08720016792550503
		features_values_sd[5]['inv_first_ISI'] = 19.856966243739745
		features_values_sd[5]['AP_fall_time'] = 1.098791663614336
		features_values_sd[5]['steady_state_voltage_stimend'] = 8.491277149856431
		features_values_sd[5]['decay_time_constant_after_stim'] = 8.243489091048744
		
		features_values_mean[6]['Spikecount'] = 10.909090909090908
		features_values_mean[6]['mean_frequency'] = 19.517475125640967
		features_values_mean[6]['AHP_depth_abs'] = -51.3987413651477
		features_values_mean[6]['AHP_depth_abs_slow'] = -53.13013173789722
		features_values_mean[6]['AHP_slow_time'] = 0.32505528946393925
		features_values_mean[6]['AP_width'] = 3.4210359463774065
		features_values_mean[6]['AP_height'] = 36.64768294376106
		features_values_mean[6]['ISI_CV'] = 0.21416195322556336
		features_values_mean[6]['inv_first_ISI'] = 48.86518384928783
		features_values_mean[6]['AP_fall_time'] = 2.066436568482347
		features_values_mean[6]['steady_state_voltage_stimend'] = -47.789257293645626
		features_values_mean[6]['decay_time_constant_after_stim'] = 12.021759309878968
		
		features_values_sd[6]['Spikecount'] = 3.5790944881871867/2
		features_values_sd[6]['mean_frequency'] = 5.256945932884365
		features_values_sd[6]['AHP_depth_abs'] = 12.22597430542713
		features_values_sd[6]['AHP_depth_abs_slow'] = 10.03929885262359
		features_values_sd[6]['AHP_slow_time'] = 0.09692947927174128
		features_values_sd[6]['AP_width'] = 2.5118115910505687
		features_values_sd[6]['AP_height'] = 11.638033103454552
		features_values_sd[6]['ISI_CV'] = 0.18541446415185062
		features_values_sd[6]['inv_first_ISI'] = 20.113669428543723
		features_values_sd[6]['AP_fall_time'] = 1.0521675325364543
		features_values_sd[6]['steady_state_voltage_stimend'] = 11.485739927731197
		features_values_sd[6]['decay_time_constant_after_stim'] = 7.00471046766083
		
		# axon features
		features_values_mean[7]['Spikecount'] = 3.727272727272727
		features_values_mean[7]['mean_frequency'] = 9.602766320585369
		features_values_mean[7]['steady_state_voltage_stimend'] = -58.40119724054735
		features_values_mean[7]['decay_time_constant_after_stim'] = 17.426882279617683
		
		features_values_sd[7]['Spikecount'] = 2.7990553306073913/2
		features_values_sd[7]['mean_frequency'] = 3.194504470538059
		features_values_sd[7]['steady_state_voltage_stimend'] = 8.590194216930989
		features_values_sd[7]['decay_time_constant_after_stim'] = 11.993943660719102
		
		features_values_mean[8]['Spikecount'] = 7.454545454545454
		features_values_mean[8]['mean_frequency'] = 18.2489821761524
		features_values_mean[8]['steady_state_voltage_stimend'] = -53.353307066898694
		features_values_mean[8]['decay_time_constant_after_stim'] = 14.601388658491764
		
		features_values_sd[8]['Spikecount'] = 4.163993632945013/2
		features_values_sd[8]['mean_frequency'] = 3.8945348914456726
		features_values_sd[8]['steady_state_voltage_stimend'] = 8.491277149856431
		features_values_sd[8]['decay_time_constant_after_stim'] = 8.243489091048744
		
		features_values_mean[9]['Spikecount'] = 10.909090909090908
		features_values_mean[9]['mean_frequency'] = 19.517475125640967
		features_values_mean[9]['steady_state_voltage_stimend'] = -47.789257293645626
		features_values_mean[9]['decay_time_constant_after_stim'] = 12.021759309878968
		
		features_values_sd[9]['Spikecount'] = 3.5790944881871867/2
		features_values_sd[9]['mean_frequency'] = 5.256945932884365
		features_values_sd[9]['steady_state_voltage_stimend'] = 11.485739927731197
		features_values_sd[9]['decay_time_constant_after_stim'] = 7.00471046766083
		
		# Construct list of all feature names
		AllFeatNames = []
		for l in features_values_mean:
			AllFeatNames.append(list(l.keys()))
		
	return list(features_values_mean), list(features_values_sd), list(AllFeatNames)

all_features, all_features_sd, AllFeatNames = get_features()
