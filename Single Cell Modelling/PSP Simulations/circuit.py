#===========================================================================
# Import, Set up MPI Variables, Load Necessary Files
#===========================================================================
from mpi4py import MPI
import time
tic_0 = time.perf_counter()
import os
from os.path import join
import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats as st
import neuron
from neuron import h, gui
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode

agegroup = 'o_rescue'

#MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
GLOBALSEED = int(sys.argv[1])

# Create new RandomState for each RANK
SEED = GLOBALSEED*10000
np.random.seed(SEED + RANK)
local_state = np.random.RandomState(SEED + RANK)
halfnorm_rv = st.halfnorm
halfnorm_rv.random_state = local_state
uniform_rv = st.uniform
uniform_rv.random_state = local_state
from net_params import *

#Mechanisms and files
print('Mechanisms found: ', os.path.isfile('mod/x86_64/special')) if RANK==0 else None
neuron.h('forall delete_section()')
neuron.load_mechanisms('mod/')
h.load_file('net_functions.hoc')
h.load_file('models/biophys_HL5PN1' + agegroup + '.hoc')
h.load_file('models/biophys_HL5MN1.hoc')
h.load_file('models/biophys_HL5BN1.hoc')
h.load_file('models/biophys_HL5VN1.hoc')


print('Importing, setting up MPI variables and loading necessary files took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

#===========================================================================
# Simulation, Analysis, and Plotting Controls
#===========================================================================
#Sim
TESTING = False # i.e.g generate 1 cell/pop, with 0.1 s runtime
no_connectivity = False

stimulate = True # Add a stimulus
MDD = False #decrease PN GtonicApic and MN2PN weight by 40%

rec_LFP = True #record LFP from center of layer
rec_DIPOLES = True #record population - wide dipoles

run_circuit_functions = True


#===========================================================================
# Params
#===========================================================================
dt = 0.025 #for both cell and network
tstart = 0.
tstop = 7000.
celsius = 34.
v_init = -80. #for both cell and network

N_cells = 1000.
N_HL5PN = int(0.70*N_cells)
N_HL5MN = int(0.15*N_cells)
N_HL5BN = int(0.10*N_cells)
N_HL5VN = int(0.05*N_cells)

cellnums = [N_HL5PN, N_HL5MN, N_HL5BN, N_HL5VN]

if TESTING:
	OUTPUTPATH = 'Circuit_output_testing'
	N_HL5PN = 1
	N_HL5MN = 1
	N_HL5BN = 1
	N_HL5VN = 1
	tstop = 100
	print('Running test...') if RANK ==0 else None

else:
	OUTPUTPATH = 'Circuit_output'
	print('Running full simulation...') if RANK==0 else None

COMM.Barrier()

networkParams = {
	'dt' : dt,
	'tstart': tstart,
	'tstop' : tstop,
	'v_init' : v_init,
	'celsius' : celsius,
	'OUTPUTPATH' : OUTPUTPATH,
	'verbose': False}

def L5depth_shift(depth):
	depth_shift = L5upper + L5PNmaxApic # shift depth to avoid PN crossings at pia
	new_depth = depth - depth_shift

	return new_depth

L5PNmaxApic = 1900. # ~max length of PN apic
L5upper = -1600. # shallow border
L5lower = -2300. # deep border

L5rangedepth = abs(L5lower-L5upper)/2 # Sets +/- range of soma positions
L5meandepth = L5upper - L5rangedepth
L5minSynLoc = L5depth_shift(L5meandepth)-L5rangedepth*2 # Sets minimum depth & range (loc to loc+scale) for synapses

both = {'section' : ['apic', 'dend'],
		'fun' : [uniform_rv, halfnorm_rv],
		'funargs' : [{'loc':L5minSynLoc, 'scale':abs(L5minSynLoc)},{'loc':L5minSynLoc, 'scale':abs(L5minSynLoc)}],
		'funweights' : [1, 1.]}
apic = {'section' : ['apic'],
		'fun' : [uniform_rv],
		'funargs' : [{'loc':L5minSynLoc, 'scale':abs(L5minSynLoc)}],
		'funweights' : [1.]}
dend = {'section' : ['dend'],
		'fun' : [uniform_rv],
		'funargs' : [{'loc':L5minSynLoc, 'scale':abs(L5minSynLoc)}],
		'funweights' : [1.]}
PNdend = {'section' : ['dend'],
		'fun' : [halfnorm_rv],
		'funargs' : [{'loc':L5minSynLoc, 'scale':abs(L5minSynLoc)}],
		'funweights' : [1.]}

pos_args = {
	'none' : dend,
	'HL5PN1HL5PN1' : both,
	'HL5PN1HL5MN1' : dend,
	'HL5PN1HL5BN1' : dend,
	'HL5PN1HL5VN1' : dend,

	'HL5MN1HL5PN1' : apic,
	'HL5MN1HL5MN1' : dend,
	'HL5MN1HL5BN1' : dend,
	'HL5MN1HL5VN1' : dend,

	'HL5BN1HL5PN1' : PNdend,
	'HL5BN1HL5MN1' : dend,
	'HL5BN1HL5BN1' : dend,
	'HL5BN1HL5VN1' : dend,

	'HL5VN1HL5PN1' : PNdend,
	'HL5VN1HL5MN1' : dend,
	'HL5VN1HL5BN1' : dend,
	'HL5VN1HL5VN1' : dend}

L5_pop_args = {'radius':250,
				'loc':L5depth_shift(L5meandepth),
				'scale':L5rangedepth*4,
				'cap': L5rangedepth}

rotations = {'HL5PN1':{'x':1.57,'y':3.72},
			 'HL5MN1':{'x':1.77,'y':2.77},
			 'HL5BN1':{'x':1.26,'y':2.57},
			 'HL5VN1':{'x':-1.57,'y':3.57}}

# class RecExtElectrode parameters:
L5_size = abs(L5upper - L5lower)
e1 = L5depth_shift(L5upper-L5_size*.5)

LFPelectrodeParameters = dict(
	x=np.zeros(1),
	y=np.zeros(1),
	z=[e1],
	N=np.array([[0., 1., 0.] for _ in range(1)]),
	r=5.,
	n=50,
	sigma=0.3,
	method="soma_as_point")


#method Network.simulate() parameters

simargs = {'rec_imem': False,
		   'rec_vmem': False,
		   'rec_ipas': False,
		   'rec_icap': False,
		   'rec_isyn': False,
		   'rec_vmemsyn': False,
		   'rec_istim': False,
		   'rec_current_dipole_moment': rec_DIPOLES,
		   'rec_pop_contributions': False,
		   'rec_variables': [],
		   'to_memory': True,
		   'to_file': False,
		   'file_name':'OUTPUT.h5',
		   'dotprodcoeffs': None}

#===========================================================================
# Functions
#===========================================================================
def generateSubPop(popsize,mname,popargs,Gou,Gtonic,GtonicApic):
	print('Initiating ' + mname + ' population...') if RANK==0 else None
	morphpath = 'morphologies/' + mname + '.swc'
	templatepath = 'models/NeuronTemplate.hoc'
	templatename = 'NeuronTemplate'
	
	pt3d = True
	
	cellParams = {
		'morphology': morphpath,
		'templatefile': templatepath,
		'templatename': templatename,
		'templateargs': morphpath,
		'v_init': v_init, #initial membrane potential, d=-65
		'passive': False,#initialize passive mechs, d=T, should be overwritten by biophys
		'dt': dt,
		'tstart': 0.,
		'tstop': tstop,#defaults to 100
		'nsegs_method': None,
		'pt3d': pt3d,#use pt3d-info of the cell geometries switch, d=F
		'delete_sections': False,
		'verbose': False}#verbose output switch, for some reason doens't work, figure out why}
	
	if mname in rotations.keys():
		rotation = rotations.get(mname)
	
	popParams = {
		'CWD': None,
		'CELLPATH': None,
		'Cell' : LFPy.NetworkCell, #play around with this, maybe put popargs into here
		'POP_SIZE': popsize,
		'name': mname,
		'cell_args' : {**cellParams},
		'pop_args' : popargs,
		'rotation_args' : rotation}
	
	network.create_population(**popParams)
	
	# Add biophys, OU processes, & tonic inhibition to cells
	for cellind in range(0,len(network.populations[mname].cells)):
		rseed = int(local_state.uniform()*SEED)
		biophys = 'h.biophys_' + mname + '(network.populations[\'' + mname + '\'].cells[' + str(cellind) + '].template)'
		exec(biophys)
		h.createArtificialSyn(rseed,network.populations[mname].cells[cellind].template,Gou)
		h.addTonicInhibition(network.populations[mname].cells[cellind].template,Gtonic,GtonicApic)

def addStimulus(stim_list, cell_nums):
	cell_names = ['HL5PN1','HL5MN1','HL5BN1','HL5VN1']
	for stim in stim_list:
		stim_index = sum(cell_nums[:cell_names.index(stim['mname'])]) + stim['num_cell'] + stim['idx_offset']
		for gid, cell in zip(network.populations[stim['mname']].gids, network.populations[stim['mname']].cells):
			if gid < stim_index:
				idx = cell.get_rand_idx_area_norm(section=stim['loc'], nidx=stim['loc_num'])
				for i in idx:
					time_d=0
					syn = Synapse(cell=cell, idx=i, syntype=stim['stim_type'], weight=1,**stim['syn_params'])
					while time_d <= 0:
						time_d = np.random.uniform(low = stim['delay'], high = stim['delay']+stim['delay_std'])
					syn.set_spike_times_w_netstim(noise=0.3, start=(stim['start_time']+time_d), number=stim['num_stim'], interval=stim['interval'], seed=GLOBALSEED)
#===========================================================================
# Sim
#===========================================================================
network = Network(**networkParams)

if MDD:
	syn_params['HL5MN1HL5PN1']['gmax'] = syn_params['HL5MN1HL5PN1']['gmax']*0.6
	syn_params['HL5MN1HL5MN1']['gmax'] = syn_params['HL5MN1HL5MN1']['gmax']*0.6
	syn_params['HL5MN1HL5BN1']['gmax'] = syn_params['HL5MN1HL5BN1']['gmax']*0.6
	syn_params['HL5MN1HL5VN1']['gmax'] = syn_params['HL5MN1HL5VN1']['gmax']*0.6
	Gtonic_MN -= (Ncont_MN2MN*con_MN2MN*connection_prob['HL5MN1HL5MN1'])/(Ncont_BN2MN*con_BN2MN*connection_prob['HL5BN1HL5MN1']+Ncont_VN2MN*con_VN2MN*connection_prob['HL5VN1HL5MN1']+Ncont_MN2MN*con_MN2MN*connection_prob['HL5MN1HL5MN1'])*Gtonic_MN*0.4
	Gtonic_BN -= (Ncont_MN2BN*con_MN2BN*connection_prob['HL5MN1HL5BN1'])/(Ncont_BN2BN*con_BN2BN*connection_prob['HL5BN1HL5BN1']+Ncont_VN2BN*con_VN2BN*connection_prob['HL5VN1HL5BN1']+Ncont_MN2BN*con_MN2BN*connection_prob['HL5MN1HL5BN1'])*Gtonic_BN*0.4
	Gtonic_VN -= (Ncont_MN2VN*con_MN2VN*connection_prob['HL5MN1HL5VN1'])/(Ncont_BN2VN*con_BN2VN*connection_prob['HL5BN1HL5VN1']+Ncont_VN2VN*con_VN2VN*connection_prob['HL5VN1HL5VN1']+Ncont_MN2VN*con_MN2VN*connection_prob['HL5MN1HL5VN1'])*Gtonic_VN*0.4
	GtonicApic_PN = Gtonic_PN*0.6
	print('MN tonic reduced by ',(Ncont_MN2MN*con_MN2MN*connection_prob['HL5MN1HL5MN1'])/(Ncont_BN2MN*con_BN2MN*connection_prob['HL5BN1HL5MN1']+Ncont_VN2MN*con_VN2MN*connection_prob['HL5VN1HL5MN1']+Ncont_MN2MN*con_MN2MN*connection_prob['HL5MN1HL5MN1'])*0.4*100, '%') if RANK==0 else None
	print('BN tonic reduced by ',(Ncont_MN2BN*con_MN2BN*connection_prob['HL5MN1HL5BN1'])/(Ncont_BN2BN*con_BN2BN*connection_prob['HL5BN1HL5BN1']+Ncont_VN2BN*con_VN2BN*connection_prob['HL5VN1HL5BN1']+Ncont_MN2BN*con_MN2BN*connection_prob['HL5MN1HL5BN1'])*0.4*100, '%') if RANK==0 else None
	print('VN tonic reduced by ',(Ncont_MN2VN*con_MN2VN*connection_prob['HL5MN1HL5VN1'])/(Ncont_BN2VN*con_BN2VN*connection_prob['HL5BN1HL5VN1']+Ncont_VN2VN*con_VN2VN*connection_prob['HL5VN1HL5VN1']+Ncont_MN2VN*con_MN2VN*connection_prob['HL5MN1HL5VN1'])*0.4*100, '%') if RANK==0 else None

# Generate Populations
tic = time.perf_counter()
generateSubPop(N_HL5PN,'HL5PN1',L5_pop_args,Gou_PN,Gtonic_PN,GtonicApic_PN)
generateSubPop(N_HL5MN,'HL5MN1',L5_pop_args,Gou_MN,Gtonic_MN,Gtonic_MN)
generateSubPop(N_HL5BN,'HL5BN1',L5_pop_args,Gou_BN,Gtonic_BN,Gtonic_BN)
generateSubPop(N_HL5VN,'HL5VN1',L5_pop_args,Gou_VN,Gtonic_VN,Gtonic_VN)
COMM.Barrier()

print('Instantiating all populations took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

tic = time.perf_counter()

# Synaptic Connection Parameters
E_syn = neuron.h.ProbAMPANMDA
I_syn = neuron.h.ProbUDFsyn

weightFunction = local_state.normal

WP = {'loc':1, 'scale':0}
MW=1


weightArguments = [[WP,WP,WP,WP],[WP,WP,WP,WP],[WP,WP,WP,WP],[WP,WP,WP,WP]]
minweightArguments = [[MW,MW,MW,MW],[MW,MW,MW,MW],[MW,MW,MW,MW],[MW,MW,MW,MW]]

delayFunction = local_state.normal
delayParams = {'loc':0.5, 'scale':0}
mindelay=0.5
delayArguments = np.full([4, 4], delayParams)

connectionProbability = [[connection_prob['HL5PN1HL5PN1'],connection_prob['HL5PN1HL5MN1'],connection_prob['HL5PN1HL5BN1'],connection_prob['HL5PN1HL5VN1']],
						 [connection_prob['HL5MN1HL5PN1'],connection_prob['HL5MN1HL5MN1'],connection_prob['HL5MN1HL5BN1'],connection_prob['HL5MN1HL5VN1']],
						 [connection_prob['HL5BN1HL5PN1'],connection_prob['HL5BN1HL5MN1'],connection_prob['HL5BN1HL5BN1'],connection_prob['HL5BN1HL5VN1']],
						 [connection_prob['HL5VN1HL5PN1'],connection_prob['HL5VN1HL5MN1'],connection_prob['HL5VN1HL5BN1'],connection_prob['HL5VN1HL5VN1']]]
if no_connectivity:
	connectionProbability = np.zeros_like(connectionProbability)

multapseFunction = local_state.normal
multapseArguments = [[mult_syns['HL5PN1HL5PN1'],mult_syns['HL5PN1HL5MN1'],mult_syns['HL5PN1HL5BN1'],mult_syns['HL5PN1HL5VN1']],
					 [mult_syns['HL5MN1HL5PN1'],mult_syns['HL5MN1HL5MN1'],mult_syns['HL5MN1HL5BN1'],mult_syns['HL5MN1HL5VN1']],
					 [mult_syns['HL5BN1HL5PN1'],mult_syns['HL5BN1HL5MN1'],mult_syns['HL5BN1HL5BN1'],mult_syns['HL5BN1HL5VN1']],
					 [mult_syns['HL5VN1HL5PN1'],mult_syns['HL5VN1HL5MN1'],mult_syns['HL5VN1HL5BN1'],mult_syns['HL5VN1HL5VN1']]]

synapseParameters = [[syn_params['HL5PN1HL5PN1'],syn_params['HL5PN1HL5MN1'],syn_params['HL5PN1HL5BN1'],syn_params['HL5PN1HL5VN1']],
					 [syn_params['HL5MN1HL5PN1'],syn_params['HL5MN1HL5MN1'],syn_params['HL5MN1HL5BN1'],syn_params['HL5MN1HL5VN1']],
					 [syn_params['HL5BN1HL5PN1'],syn_params['HL5BN1HL5MN1'],syn_params['HL5BN1HL5BN1'],syn_params['HL5BN1HL5VN1']],
					 [syn_params['HL5VN1HL5PN1'],syn_params['HL5VN1HL5MN1'],syn_params['HL5VN1HL5BN1'],syn_params['HL5VN1HL5VN1']]]

synapsePositionArguments = [[pos_args['HL5PN1HL5PN1'],pos_args['HL5PN1HL5MN1'],pos_args['HL5PN1HL5BN1'],pos_args['HL5PN1HL5VN1']],
							[pos_args['HL5MN1HL5PN1'],pos_args['HL5MN1HL5MN1'],pos_args['HL5MN1HL5BN1'],pos_args['HL5MN1HL5VN1']],
							[pos_args['HL5BN1HL5PN1'],pos_args['HL5BN1HL5MN1'],pos_args['HL5BN1HL5BN1'],pos_args['HL5BN1HL5VN1']],
							[pos_args['HL5VN1HL5PN1'],pos_args['HL5VN1HL5MN1'],pos_args['HL5VN1HL5BN1'],pos_args['HL5VN1HL5VN1']]]

for i, pre in enumerate(network.population_names):
	for j, post in enumerate(network.population_names):

		connectivity = network.get_connectivity_rand(
			pre=pre,
			post=post,
			connprob=connectionProbability[i][j])

		(conncount, syncount) = network.connect(
			pre=pre, post=post,
			connectivity=connectivity,
			syntype=E_syn if pre=='HL5PN1' else I_syn,
			synparams=synapseParameters[i][j],
			weightfun=weightFunction,
			weightargs=weightArguments[i][j],
			minweight=minweightArguments[i][j],
			delayfun=delayFunction,
			delayargs=delayArguments[i][j],
			mindelay=mindelay,
			multapsefun=multapseFunction,
			multapseargs=multapseArguments[i][j],
			syn_pos_args=synapsePositionArguments[i][j])

print('Connecting populations took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

# Setup Extracellular Recording Device
COMM.Barrier()
if stimulate:
	addStimulus(cells_to_stim, cellnums)
COMM.Barrier()


# Run Simulation

tic = time.perf_counter()
LFPelectrode = RecExtElectrode(**LFPelectrodeParameters) if rec_LFP else None

if rec_LFP and not rec_DIPOLES:
	print('Simulating, recording SPIKES and LFP ... ') if RANK==0 else None
	SPIKES, OUTPUT = network.simulate(electrode=LFPelectrode,**simargs)
elif rec_LFP and rec_DIPOLES:
	print('Simulating, recording SPIKES, LFP, and DIPOLEMOMENTS ... ') if RANK==0 else None
	SPIKES, OUTPUT, DIPOLEMOMENT = network.simulate(electrode=LFPelectrode,**simargs)
elif rec_DIPOLES and not rec_LFP:
	print('Simulating, recording SPIKES DIPOLEMOMENTS ... ') if RANK==0 else None
	SPIKES, DIPOLEMOMENT = network.simulate(**simargs)
elif not rec_LFP and not rec_DIPOLES:
	print('Simulating, recording SPIKES ... ') if RANK==0 else None
	SPIKES = network.simulate(**simargs)

print('Simulation took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None


COMM.Barrier()
if RANK==0:
	tic = time.perf_counter()
	print('Saving simulation output...')
	np.save(os.path.join(OUTPUTPATH,'SPIKES_Seed'+str(GLOBALSEED)+'.npy'), SPIKES)
	np.save(os.path.join(OUTPUTPATH,'OUTPUT_Seed'+str(GLOBALSEED)+'.npy'), OUTPUT) if rec_LFP else None
	np.save(os.path.join(OUTPUTPATH,'DIPOLEMOMENT_Seed'+str(GLOBALSEED)+'.npy'), DIPOLEMOMENT) if rec_DIPOLES else None
	print('Saving simulation took', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')


#===========================================================================
# Plotting
#===========================================================================
if run_circuit_functions:

	tstart_plot = 2000
	tstop_plot = tstop

	print('Creating/saving plots...') if RANK==0 else None
	exec(open("circuit_functions.py").read())

#===============
#Final printouts
#===============
if TESTING:
	print('Test complete, switch TESTING to False for full simulation') if RANK==0 else None
elif not TESTING:
	print('Simulation complete') if RANK==0 else None

print('Script completed in ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
