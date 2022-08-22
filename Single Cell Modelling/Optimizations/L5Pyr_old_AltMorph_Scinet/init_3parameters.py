##############################################
### Setup Parameters Values and Boundaries ###
##############################################

######### Frozen Passive Parameters #########
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
cm_axonal_param = ephys.parameters.NrnSectionParameter(
	name ='cmaxonal',
	param_name ='cm',
	locations = [axonal_loc],
	value = 0.9,
	frozen = True
)
if cellname == 'pyramidal':
	cm_apical_param = ephys.parameters.NrnSectionParameter(
		name ='cmapical',
		param_name ='cm',
		locations = [apical_loc],
		value = 0.9,
		frozen = True
	)

######### Free Passive Parameters #########
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
if cellname == 'pyramidal':
	# Scales IH with distance from soma
	scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(name='Ihscaler',distribution="(0.5 + 24/(1 + math.exp(({distance} - 950)/-285))) * {value}") # Absolute Sigmoidal

act_params={}
for x in active_params:
	if x['section'] == 'somatic':
		loc = somatic_loc
	elif x['section'] == 'basal':
		loc = basal_loc
	elif x['section'] == 'apical':
		loc = apical_loc
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
		if cellname == 'pyramidal': # Add nonlinear distance-dependence if pyramidal
			act_params["{0}{1}_param".format(x['name'],x['section'])]=ephys.parameters.NrnRangeParameter(
				name =x['name']+x['section'],
				param_name =x['name'],
				value_scaler=scaler,
				locations = [loc],
				value = 0,
				bounds=[0, 0.0001],
				frozen = False)
		else:
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

######### Additional Frozen Active Parameters #########
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

# Construct list of all parameters (inputted to Cell_Model below)
if cellname == 'pyramidal':
	mod_params = [celsius_param, v_init_param, ena_paramsomatic, ena_paramaxonal, ek_paramsomatic, ek_paramaxonal, e_pas_param, Ra_param, cm_somatic_param, cm_basal_param, cm_apical_param, cm_axonal_param, g_pas_param]
else:
	mod_params = [celsius_param, v_init_param, ena_paramsomatic, ena_paramaxonal, ek_paramsomatic, ek_paramaxonal, e_pas_param, Ra_param, cm_somatic_param, cm_basal_param, cm_axonal_param, g_pas_param]

for name in act_params: mod_params.append(act_params[name])

# Construct list of free parameters (inputted to fitness calculator later on)
nonfrozen_params=['g_pas','e_pas'] # first enter free parameters not found in the active_paras list
for x in active_params:
	if (x['name'] =='gamma_CaDynamics'): # i.e. since this parameter is frozen
		continue
	else:
		nonfrozen_params.append(x['name']+x['section'])

######### Create Model #########

Cell_Model = ephys.models.CellModel(
	name = cellname,
	morph = morphology,
	mechs = mod_mechs,
	params = mod_params
)
