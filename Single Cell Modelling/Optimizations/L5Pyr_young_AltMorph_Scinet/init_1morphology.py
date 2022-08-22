########################
### Setup Morphology ###
########################

######### Load Morphology #########

cellname = 'pyramidal' # options: 'pyramidal' or 'interneuron' - select 'pyramidal' if morphology contains apical dendrites
morphname = 'experimental_data/H16.06.010.01.03.05.02_599474744_m.swc'
CustomAxonReplacement = False # Patches the replace_axon function in bluepyopt with the function defined below

# Replace axon method - Note the python and hoc replace_axon functions need to be equivalent for correct hoc model exports (see out_2SimulateHocModel.py)
def replace_axon(sim=None, icell=None):
	import numpy
	for section in icell.axonal:
		sim.neuron.h.delete_section(sec=section)
	
	# Create new axon array
	sim.neuron.h.execute('create axon[2]', icell)
	
	icell.axon[0].L = 20
	icell.axon[0].nseg = 1+2*int(icell.axon[0].L/10)
	dx = 1./icell.axon[0].nseg
	zero_bound = 3
	one_bound = 1.75
	for (seg, x) in zip(icell.axon[0], numpy.arange(dx/2, 1, dx)):
		seg.diam=(one_bound-zero_bound)*x+zero_bound
	icell.axonal.append(sec=icell.axon[0])
	icell.all.append(sec=icell.axon[0])
	
	icell.axon[1].L = 30
	icell.axon[1].nseg = 1+2*int(icell.axon[1].L/10)
	dx = 1./icell.axon[1].nseg
	zero_bound = 1.75
	one_bound = 1
	for (seg, x) in zip(icell.axon[1], numpy.arange(dx/2, 1, dx)):
		seg.diam=(one_bound-zero_bound)*x+zero_bound
	icell.axonal.append(sec=icell.axon[1])
	icell.all.append(sec=icell.axon[1])
	
	icell.axon[0].connect(icell.soma[0], 0.5, 0.0)
	icell.axon[1].connect(icell.axon[0], 1.0, 0.0)
	
	# test = test2 # bug

replace_axon_hoc = \
'''
proc replace_axon(){
	forsec axonal{delete_section()}
	create axon[2]
	access axon[0]{
		L= 20
		nseg = 1+2*int(L/10)
		diam(0:1) = 3:1.75
		all.append()
		axonal.append()
	}
	access axon[1]{
		L= 30
		nseg = 1+2*int(L/10)
		diam(0:1) = 1.75:1
		all.append()
		axonal.append()
	}
	
	nSecAxonal = 2
	connect axon(0), soma(0.5)
	connect axon[1](0), axon[0](1)
	access soma
}
'''

# Overwrite replace_axon function with monkey patch
if CustomAxonReplacement:
	ephys.morphologies.NrnFileMorphology.replace_axon = staticmethod(replace_axon)

morphology = ephys.morphologies.NrnFileMorphology(morphname, do_replace_axon=True, replace_axon_hoc=replace_axon_hoc if CustomAxonReplacement else None)

all_loc = ephys.locations.NrnSeclistLocation('all', seclist_name='all')
somatic_loc = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')
basal_loc = ephys.locations.NrnSeclistLocation('basal', seclist_name='basal')
axonal_loc = ephys.locations.NrnSeclistLocation('axonal', seclist_name='axonal')
if cellname == 'pyramidal':
	apical_loc = ephys.locations.NrnSeclistLocation('apical', seclist_name='apical')
