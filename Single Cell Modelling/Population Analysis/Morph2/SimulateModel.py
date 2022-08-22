######### Relevent Links #########
# https://nbviewer.jupyter.org/github/BlueBrain/SimulationTutorials/blob/master/FENS2016/ABI_model/single_cell_model.ipynb
import time
import numpy
import matplotlib.pyplot as plt
from neuron import h, gui
import bluepyopt as bpop
import bluepyopt.ephys as ephys
import json
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

######### Load Morphology and Model Data #########
cellname = 'PutativePN'
morphname = 'HL5PN1.swc'

morphology = ephys.morphologies.NrnFileMorphology(morphname, do_replace_axon=True)
all_loc = ephys.locations.NrnSeclistLocation('all', seclist_name='all')
somatic_loc = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')
basal_loc = ephys.locations.NrnSeclistLocation('basal', seclist_name='basal')
apical_loc = ephys.locations.NrnSeclistLocation('apical', seclist_name='apical')
axonal_loc = ephys.locations.NrnSeclistLocation('axonal', seclist_name='axonal')

## Delete some channels and change other channels to be consistent with Hay Lab channels
## Also delete some dendritic/axonal channels
active_soma_params1 = []
param = {}
param['section'] = 'somatic'
param['name'] = 'gbar_NaTg'
param['mechanism'] = 'NaTg'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Nap'
param['mechanism'] = 'Nap'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_K_P'
param['mechanism'] = 'K_P'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_K_T'
param['mechanism'] = 'K_T'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Kv3_1'
param['mechanism'] = 'Kv3_1'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Im'
param['mechanism'] = 'Im'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_SK'
param['mechanism'] = 'SK'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'decay_CaDynamics'
param['mechanism'] = 'CaDynamics'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gamma_CaDynamics'
param['mechanism'] = 'CaDynamics'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Ca_HVA'
param['mechanism'] = 'Ca_HVA'
active_soma_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Ca_LVA'
param['mechanism'] = 'Ca_LVA'
active_soma_params1.append(copy.deepcopy(param))
param['section'] = 'all'
param['name'] = 'gbar_Ih'
param['mechanism'] = 'Ih'
active_soma_params1.append(copy.deepcopy(param))

active_axon_params1 = []
param = {}
param['section'] = 'axonal'
param['name'] = 'gbar_NaTg'
param['mechanism'] = 'NaTg'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Nap'
param['mechanism'] = 'Nap'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_K_P'
param['mechanism'] = 'K_P'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_K_T'
param['mechanism'] = 'K_T'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Kv3_1'
param['mechanism'] = 'Kv3_1'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Im'
param['mechanism'] = 'Im'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_SK'
param['mechanism'] = 'SK'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'decay_CaDynamics'
param['mechanism'] = 'CaDynamics'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gamma_CaDynamics'
param['mechanism'] = 'CaDynamics'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Ca_HVA'
param['mechanism'] = 'Ca_HVA'
active_axon_params1.append(copy.deepcopy(param))
param['name'] = 'gbar_Ca_LVA'
param['mechanism'] = 'Ca_LVA'
active_axon_params1.append(copy.deepcopy(param))

active_params = []
active_params.extend(active_soma_params1)
active_params.extend(active_axon_params1)

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
		
		features_values1[0]['sag_amplitude'] = 5.278100726466888
		features_values1[0]['sag_ratio1'] = 0.1481642228342842
		features_values1[0]['ohmic_input_resistance_vb_ssse'] = 74.03527452374264
		features_values1[0]['voltage_base'] = -70.88430730663849
		
		features_values2[0]['sag_amplitude_std'] = 2.6660693952549757/10
		features_values2[0]['sag_ratio1_std'] = 0.04456314652499393
		features_values2[0]['ohmic_input_resistance_vb_ssse_std'] = 24.36723464645282
		features_values2[0]['voltage_base_std'] = 4.847965941570795/2
		
		features_values1[1]['sag_amplitude'] = 4.14297392783007
		features_values1[1]['sag_ratio1'] = 0.1430363798705737
		features_values1[1]['ohmic_input_resistance_vb_ssse'] = 80.30606348785402
		features_values1[1]['voltage_base'] = -70.77824081894033
		
		features_values2[1]['sag_amplitude_std'] = 2.4048490063182313/10
		features_values2[1]['sag_ratio1_std'] = 0.05418255221975284
		features_values2[1]['ohmic_input_resistance_vb_ssse_std'] = 26.22398634726206
		features_values2[1]['voltage_base_std'] = 5.050300427120481/2
		
		features_values1[2]['sag_amplitude'] = 3.0893530826949833
		features_values1[2]['sag_ratio1'] = 0.14641767702365857
		features_values1[2]['ohmic_input_resistance_vb_ssse'] = 89.28021086034806
		features_values1[2]['voltage_base'] = -70.75919314447394
		
		features_values2[2]['sag_amplitude_std'] = 1.7988838813077013/10
		features_values2[2]['sag_ratio1_std'] = 0.058254740064004454
		features_values2[2]['ohmic_input_resistance_vb_ssse_std'] = 30.009528125222737
		features_values2[2]['voltage_base_std'] = 5.234119677048471/2
		
		features_values1[3]['sag_amplitude'] = 1.5375874896612003
		features_values1[3]['sag_ratio1'] = 0.22016542016166685
		features_values1[3]['ohmic_input_resistance_vb_ssse'] = 110.4504308246921
		features_values1[3]['voltage_base'] = -71.03105667954445
		
		features_values2[3]['sag_amplitude_std'] = 0.850533526773676/5
		features_values2[3]['sag_ratio1_std'] = 0.09671918252213504
		features_values2[3]['ohmic_input_resistance_vb_ssse_std'] = 46.54981874949187
		features_values2[3]['voltage_base_std'] = 4.800949900675258/2
		
		features_values1[4]['Spikecount'] = 3.727272727272727
		features_values1[4]['mean_frequency'] = 9.602766320585369
		features_values1[4]['AHP_depth_abs'] = -58.07434994377865
		features_values1[4]['AHP_depth_abs_slow'] = -60.39359192619093
		features_values1[4]['AHP_slow_time'] = 0.2649655033059525
		features_values1[4]['AP_width'] = 2.4566666666670987
		features_values1[4]['AP_height'] = 44.47247679980026
		features_values1[4]['ISI_CV'] = 0.17641360052825844
		features_values1[4]['inv_first_ISI'] = 11.19333032135457
		features_values1[4]['AP_fall_time'] = 2.008958333333663
		features_values1[4]['steady_state_voltage_stimend'] = -58.40119724054735
		features_values1[4]['decay_time_constant_after_stim'] = 17.426882279617683
		
		features_values2[4]['Spikecount_std'] = 2.7990553306073913
		features_values2[4]['mean_frequency_std'] = 3.194504470538059
		features_values2[4]['AHP_depth_abs_std'] = 6.353539415690842
		features_values2[4]['AHP_depth_abs_slow_std'] = 5.809293847391061
		features_values2[4]['AHP_slow_time_std'] = 0.09423464665336653
		features_values2[4]['AP_width_std'] = 0.8745121655975552
		features_values2[4]['AP_height_std'] = 7.394850554959106
		features_values2[4]['ISI_CV_std'] = 0.13409875646853456
		features_values2[4]['inv_first_ISI_std'] = 11.588141398703439
		features_values2[4]['AP_fall_time_std'] = 1.1452495293059304
		features_values2[4]['steady_state_voltage_stimend_std'] = 8.590194216930989
		features_values2[4]['decay_time_constant_after_stim_std'] = 11.993943660719102
		
		features_values1[5]['Spikecount'] = 7.454545454545454
		features_values1[5]['mean_frequency'] = 18.2489821761524
		features_values1[5]['AHP_depth_abs'] = -55.13927279609413
		features_values1[5]['AHP_depth_abs_slow'] = -56.73911699122267
		features_values1[5]['AHP_slow_time'] = 0.3007859939045979
		features_values1[5]['AP_width'] = 2.823222222222639
		features_values1[5]['AP_height'] = 38.981035646708285
		features_values1[5]['ISI_CV'] = 0.15304857074487668
		features_values1[5]['inv_first_ISI'] = 32.339931952506134
		features_values1[5]['AP_fall_time'] = 2.0331047008549485
		features_values1[5]['steady_state_voltage_stimend'] = -53.353307066898694
		features_values1[5]['decay_time_constant_after_stim'] = 14.601388658491764
		
		features_values2[5]['Spikecount_std'] = 4.163993632945013/2
		features_values2[5]['mean_frequency_std'] = 3.8945348914456726
		features_values2[5]['AHP_depth_abs_std'] = 8.088493834520389
		features_values2[5]['AHP_depth_abs_slow_std'] = 8.089093654518868
		features_values2[5]['AHP_slow_time_std'] = 0.10255675046087279
		features_values2[5]['AP_width_std'] = 1.4785934632719298
		features_values2[5]['AP_height_std'] = 9.283539200158264
		features_values2[5]['ISI_CV_std'] = 0.08720016792550503
		features_values2[5]['inv_first_ISI_std'] = 19.856966243739745
		features_values2[5]['AP_fall_time_std'] = 1.098791663614336
		features_values2[5]['steady_state_voltage_stimend_std'] = 8.491277149856431
		features_values2[5]['decay_time_constant_after_stim_std'] = 8.243489091048744
		
		features_values1[6]['Spikecount'] = 10.909090909090908
		features_values1[6]['mean_frequency'] = 19.517475125640967
		features_values1[6]['AHP_depth_abs'] = -51.3987413651477
		features_values1[6]['AHP_depth_abs_slow'] = -53.13013173789722
		features_values1[6]['AHP_slow_time'] = 0.32505528946393925
		features_values1[6]['AP_width'] = 3.4210359463774065
		features_values1[6]['AP_height'] = 36.64768294376106
		features_values1[6]['ISI_CV'] = 0.21416195322556336
		features_values1[6]['inv_first_ISI'] = 48.86518384928783
		features_values1[6]['AP_fall_time'] = 2.066436568482347
		features_values1[6]['steady_state_voltage_stimend'] = -47.789257293645626
		features_values1[6]['decay_time_constant_after_stim'] = 12.021759309878968
		
		features_values2[6]['Spikecount_std'] = 3.5790944881871867/2
		features_values2[6]['mean_frequency_std'] = 5.256945932884365
		features_values2[6]['AHP_depth_abs_std'] = 12.22597430542713
		features_values2[6]['AHP_depth_abs_slow_std'] = 10.03929885262359
		features_values2[6]['AHP_slow_time_std'] = 0.09692947927174128
		features_values2[6]['AP_width_std'] = 2.5118115910505687
		features_values2[6]['AP_height_std'] = 11.638033103454552
		features_values2[6]['ISI_CV_std'] = 0.18541446415185062
		features_values2[6]['inv_first_ISI_std'] = 20.113669428543723
		features_values2[6]['AP_fall_time_std'] = 1.0521675325364543
		features_values2[6]['steady_state_voltage_stimend_std'] = 11.485739927731197
		features_values2[6]['decay_time_constant_after_stim_std'] = 7.00471046766083
		
		features_values1[7]['Spikecount'] = 3.727272727272727
		features_values1[7]['mean_frequency'] = 9.602766320585369
		# features_values1[7]['AHP_depth_abs'] = -58.07434994377865
		# features_values1[7]['AHP_depth_abs_slow'] = -60.39359192619093
		# features_values1[7]['AHP_slow_time'] = 0.2649655033059525
		# features_values1[7]['AP_width'] = 2.4566666666670987
		# features_values1[7]['AP_height'] = 44.47247679980026
		# features_values1[7]['ISI_CV'] = 0.17641360052825844
		# features_values1[7]['inv_first_ISI'] = 11.19333032135457
		# features_values1[7]['AP_fall_time'] = 2.008958333333663
		features_values1[7]['steady_state_voltage_stimend'] = -58.40119724054735
		features_values1[7]['decay_time_constant_after_stim'] = 17.426882279617683
		
		features_values2[7]['Spikecount_std'] = 2.7990553306073913/2
		features_values2[7]['mean_frequency_std'] = 3.194504470538059
		# features_values2[7]['AHP_depth_abs_std'] = 6.353539415690842
		# features_values2[7]['AHP_depth_abs_slow_std'] = 5.809293847391061
		# features_values2[7]['AHP_slow_time_std'] = 0.09423464665336653
		# features_values2[7]['AP_width_std'] = 0.8745121655975552
		# features_values2[7]['AP_height_std'] = 7.394850554959106
		# features_values2[7]['ISI_CV_std'] = 0.13409875646853456
		# features_values2[7]['inv_first_ISI_std'] = 11.588141398703439
		# features_values2[7]['AP_fall_time_std'] = 1.1452495293059304
		features_values2[7]['steady_state_voltage_stimend_std'] = 8.590194216930989
		features_values2[7]['decay_time_constant_after_stim_std'] = 11.993943660719102
		
		features_values1[8]['Spikecount'] = 7.454545454545454
		features_values1[8]['mean_frequency'] = 18.2489821761524
		# features_values1[8]['AHP_depth_abs'] = -55.13927279609413
		# features_values1[8]['AHP_depth_abs_slow'] = -56.73911699122267
		# features_values1[8]['AHP_slow_time'] = 0.3007859939045979
		# features_values1[8]['AP_width'] = 2.823222222222639
		# features_values1[8]['AP_height'] = 38.981035646708285
		# features_values1[8]['ISI_CV'] = 0.15304857074487668
		# features_values1[8]['inv_first_ISI'] = 32.339931952506134
		# features_values1[8]['AP_fall_time'] = 2.0331047008549485
		features_values1[8]['steady_state_voltage_stimend'] = -53.353307066898694
		features_values1[8]['decay_time_constant_after_stim'] = 14.601388658491764
		
		features_values2[8]['Spikecount_std'] = 4.163993632945013/2
		features_values2[8]['mean_frequency_std'] = 3.8945348914456726
		# features_values2[8]['AHP_depth_abs_std'] = 8.088493834520389
		# features_values2[8]['AHP_depth_abs_slow_std'] = 8.089093654518868
		# features_values2[8]['AHP_slow_time_std'] = 0.10255675046087279
		# features_values2[8]['AP_width_std'] = 1.4785934632719298
		# features_values2[8]['AP_height_std'] = 9.283539200158264
		# features_values2[8]['ISI_CV_std'] = 0.08720016792550503
		# features_values2[8]['inv_first_ISI_std'] = 19.856966243739745
		# features_values2[8]['AP_fall_time_std'] = 1.098791663614336
		features_values2[8]['steady_state_voltage_stimend_std'] = 8.491277149856431
		features_values2[8]['decay_time_constant_after_stim_std'] = 8.243489091048744
		
		features_values1[9]['Spikecount'] = 10.909090909090908
		features_values1[9]['mean_frequency'] = 19.517475125640967
		# features_values1[9]['AHP_depth_abs'] = -51.3987413651477
		# features_values1[9]['AHP_depth_abs_slow'] = -53.13013173789722
		# features_values1[9]['AHP_slow_time'] = 0.32505528946393925
		# features_values1[9]['AP_width'] = 3.4210359463774065
		# features_values1[9]['AP_height'] = 36.64768294376106
		# features_values1[9]['ISI_CV'] = 0.21416195322556336
		# features_values1[9]['inv_first_ISI'] = 48.86518384928783
		# features_values1[9]['AP_fall_time'] = 2.066436568482347
		features_values1[9]['steady_state_voltage_stimend'] = -47.789257293645626
		features_values1[9]['decay_time_constant_after_stim'] = 12.021759309878968
		
		features_values2[9]['Spikecount_std'] = 3.5790944881871867/2
		features_values2[9]['mean_frequency_std'] = 5.256945932884365
		# features_values2[9]['AHP_depth_abs_std'] = 12.22597430542713
		# features_values2[9]['AHP_depth_abs_slow_std'] = 10.03929885262359
		# features_values2[9]['AHP_slow_time_std'] = 0.09692947927174128
		# features_values2[9]['AP_width_std'] = 2.5118115910505687
		# features_values2[9]['AP_height_std'] = 11.638033103454552
		# features_values2[9]['ISI_CV_std'] = 0.18541446415185062
		# features_values2[9]['inv_first_ISI_std'] = 20.113669428543723
		# features_values2[9]['AP_fall_time_std'] = 1.0521675325364543
		features_values2[9]['steady_state_voltage_stimend_std'] = 11.485739927731197
		features_values2[9]['decay_time_constant_after_stim_std'] = 7.00471046766083
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
	bounds=[0.00000001, 0.0002],
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
# Need to import ipyparallel for bluepyopt to run
from ipyparallel import Client
# rc = Client(profile=os.getenv('IPYTHON_PROFILE'))
#
# sys.stderr.write('Using ipyparallel with {0} engines'.format(len(rc)))
# sys.stderr.flush()
# sys.stderr.write("\n")
# sys.stderr.flush()
#
# lview = rc.load_balanced_view()
#
# def mapper(func, it):
# 	start_time = time.time()
# 	ret = lview.map_async(func, it)
# 	sys.stderr.write('Generation took {0}'.format(time.time() - start_time))
# 	sys.stderr.flush()
# 	sys.stderr.write("\n")
# 	sys.stderr.flush()
# 	return ret
#
# map_function = mapper

#################
### Run Model ###
#################


dirpath1 = 'young'

finalpop_path = dirpath1+'/finalpop.pkl'
halloffame_path = dirpath1+'/halloffame.pkl'
hist_path = dirpath1+'/hist.pkl'
logs_path = dirpath1+'/logs.pkl'

finalpop = pickle.load(open(finalpop_path,"rb"))
halloffame = pickle.load(open(halloffame_path,"rb"))
hist = pickle.load(open(hist_path,"rb"))
logs = pickle.load(open(logs_path,"rb"))

##### Print Top Models #####
idx = 0
best_ind = halloffame[idx]

# sys.stdout.flush()
# print('Best #' + str(idx+1) + ': ')
# pp.pprint(list(zip(nonfrozen_params_act,best_ind)))
# sys.stdout.flush()
# print('Best #' + str(idx+1) + ' Fitness values: ')
# pp.pprint(sum(best_ind.fitness.values))
# sys.stdout.flush()
text_file = open('results/' + dirpath1 + 'N' + str(idx+1) + 'params.txt', "w")
text_file.write('\nBest #' + str(idx+1) + ': \n')
text_file.write(pp.pformat(list(zip(nonfrozen_params_act,best_ind))))
text_file.write('\nBest #' + str(idx+1) + ' Fitness values: \n')
text_file.write(str(sum(best_ind.fitness.values)))

for c, bi in enumerate(halloffame):
	txtf = open('results/biophys_HL5PN1' + dirpath1[0] + str(c+1) + '.hoc', "w")
	txtf.write('proc biophys_HL5PN1(){\n')
	txtf.write('	forsec $o1.all {\n')
	txtf.write('		insert pas\n')
	txtf.write('		insert Ih\n')
	txtf.write('		Ra = 100\n')
	txtf.write('		cm = 0.9\n')
	txtf.write('		e_pas = ' + str(bi[1]) + '\n')
	txtf.write('		g_pas = ' + str(bi[0]) + '\n')
	txtf.write('		gbar_Ih = ' + str(bi[12]) + '\n')
	txtf.write('		shift1_Ih = 144.76545935424588\n')
	txtf.write('		shift2_Ih = 14.382865335237211\n')
	txtf.write('		shift3_Ih = -28.179477866349245\n')
	txtf.write('		shift4_Ih = 99.18311385307702\n')
	txtf.write('		shift5_Ih = 16.42000098505615\n')
	txtf.write('		shift6_Ih = 26.699880497099517\n')
	txtf.write('	}\n')
	txtf.write('	$o1.distribute_channels("apic","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("dend","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("axon","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	\n')
	txtf.write('	forsec $o1.somatic {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[2]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[3]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[4]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[5]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[6]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[8]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[7]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[10]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[11]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[9]) + '\n')
	txtf.write('	}\n')
	txtf.write('	forsec $o1.axonal {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[13]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[14]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[15]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[16]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[17]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[19]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[18]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[21]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[22]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[20]) + '\n')
	txtf.write('	}\n')
	txtf.write('}\n')
	txtf.write('\n')

best_ind_dict = cell_evaluator_act.param_dict(best_ind)

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
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_pas.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_pas.png', bbox_inches='tight', dpi=300, transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act(responses)
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_act.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_act.png', bbox_inches='tight', dpi=300, transparent=True)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
plot_responses_act2(responses)
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_act2.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('PLOTfiles/' + dirpath1 + '_OptimizedTraces' + str(idx+1) + '_act2.png', bbox_inches='tight', dpi=300, transparent=True)
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

# print("\nFeatures:\n")
# sys.stdout.flush()
# pp.pprint(active_features1)
# sys.stdout.flush()

active_errs1 = get_stdevs(active_features1)

# print("\nErrors:\n")
# sys.stdout.flush()
# pp.pprint(active_errs1)
# sys.stdout.flush()

active_steps2 = [str(s) + ' nA' for s in active_steps]
active_steps2[-1] = active_steps2[-1] + ' (axon)'
active_steps2[-2] = active_steps2[-2] + ' (axon)'
active_steps2[-3] = active_steps2[-3] + ' (axon)'

text_file.write('\nFeatures:\n')
text_file.write(pp.pformat(list(zip(active_steps2,active_features1))))
text_file.write('\nErrors (not weight compensated):\n')
text_file.write(pp.pformat(list(zip(active_steps2,active_errs1))))
