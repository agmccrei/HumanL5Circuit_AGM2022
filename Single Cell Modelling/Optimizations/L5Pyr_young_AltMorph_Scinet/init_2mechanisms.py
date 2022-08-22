#############################
### Setup Mechanism Lists ###
#############################

## Create dict of mechanisms
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

######### Create Mod Mechanism Dictionary #########
all_mechs={}
for x in active_params:
	if x['section'] == 'somatic':
		loc = somatic_loc
	elif x['section'] == 'basal':
		loc = basal_loc
	elif x['section'] == 'axonal':
		loc = axonal_loc
	elif x['section'] == 'apical':
		loc = apical_loc
	elif x['section'] == 'all':
		loc = all_loc
	all_mechs["{0}{1}_mech".format(x['mechanism'],x['section'])]=ephys.mechanisms.NrnMODMechanism(
		name = x['mechanism']+x['section'],
		suffix = x['mechanism'],
		locations = [loc])

# Also add in passive
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

all_mechs['passomatic_mech'] = passomatic_mech
all_mechs['pasbasal_mech'] = pasbasal_mech
all_mechs['pasaxonal_mech'] = pasaxonal_mech

if cellname == 'pyramidal':
	pasapical_mech = ephys.mechanisms.NrnMODMechanism(
		name ='pasapical',
		suffix ='pas',
		locations = [apical_loc]
	)
	all_mechs['pasapical_mech'] = pasapical_mech

# Create list of all mechanisms
mod_mechs = []
for name in all_mechs: mod_mechs.append(all_mechs[name])
