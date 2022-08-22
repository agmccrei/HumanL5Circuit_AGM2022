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
morphname = 'H16.06.013.11.18.02_696537235_m.swc'

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
pp.pprint(active_params)
sys.stdout.flush()

if sys.version_info[0] == 2:
	execfile("init_active.py")
elif sys.version_info[0] == 3:
	exec(open("init_active.py").read())

TotalTime = time.time() - TotalStartTime
print("Total Time = " + str(TotalTime) + " seconds")

quit()
