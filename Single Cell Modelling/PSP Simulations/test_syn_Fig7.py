import os
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import scipy
from scipy import stats as st
from mpi4py import MPI
import neuron
from neuron import *
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode
from net_params import *
from re import search
from currents_visualization import plotCurrentscape
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import pandas as pd

#MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
GLOBALSEED = 1234
np.random.seed(GLOBALSEED*2 + RANK)

################################################################################
# Simulation & Analysis Controls
################################################################################
dt = 0.025
tmid = 500 #beginning of stimulation
tstop = 850.
celsius = 34.
v_init = -68

#for visualization
startslice = 450
endslice = 730

N_connection_iterations = 20
N_trial_iterations = 20

#set both to false to run both conditions, script runs in all 3 conditions
compensate = True # compensate for difference in RMP
shifttrace = False # shift trace when plotting (if also not compensating)
PlotMorphs = True
bootCI_trials = False
bootCI_seeds = True

numpulses = 5
tfin = 600 # 600 for 50 Hz with 5 pulses
pulsefreq = numpulses/((tfin-tmid)/1000)
train_delay = np.arange(tmid, tfin, (1/pulsefreq)*1000)
step_instead_of_train = False # Applies square step instead of train of pulses

agegroups = ['y','o']
preNs = ['HL5PN1','HL5MN1','HL5BN1']
postN = 'HL5PN1'

post_somav_areas = [[[] for _ in agegroups] for _ in preNs]
post_somav_amplitude = [[[] for _ in agegroups] for _ in preNs]
post_somav_mconnection_traces = [[[] for _ in agegroups] for _ in preNs]

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

neuron.h('forall delete_section()')
print('Mechanisms found: ' + str(os.path.isfile('mod/x86_64/special')), '\n')
neuron.load_mechanisms('mod/')

if PlotMorphs:
	fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(25,18),sharex=False,sharey=False,frameon=False)
fig2, ax2 = plt.subplots(nrows=3,ncols=1,figsize=(7,15),sharex=True,sharey=False,frameon=False)
fig3, ax3 = plt.subplots(nrows=1,ncols=2,figsize=(17,9),sharex=True,sharey=True,frameon=False)

for aidx, agegroup in enumerate(agegroups):
	
	# Same seeds for young and old
	np.random.seed(GLOBALSEED*2 + RANK)
	local_state = np.random.RandomState(GLOBALSEED + RANK)
	halfnorm_rv = st.halfnorm
	halfnorm_rv.random_state = local_state
	uniform_rv = st.uniform
	uniform_rv.random_state = local_state
	
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
	
	# Postsynaptic protocol parameters
	if agegroup == 'y':
		col1 = 'k'
		col2 = 'dimgray'
		if compensate:
			postN_amp = 0.00665
		else:
			postN_amp = 0
		postN_dur = tstop
	elif agegroup == 'o':
		col1 = 'r'
		col2 = 'lightcoral'
		if compensate:
			postN_amp = -0.00665
		else:
			postN_amp = 0
		postN_dur = tstop
	
	for pidx, preN in enumerate(preNs):
		neuron.h('forall delete_section()')
		
		# Presynaptic protocol parameters
		if preN == 'HL5PN1':
			preN_amp_step = 0.30 # amplitude of square current step
			preN_dur_step = 150 # duration of square current step
			preN_amp = 3
			preN_dur = 2
		elif preN == 'HL5MN1':
			preN_amp_step = 0.25 # amplitude of square current step
			preN_dur_step = 150 # duration of square current step
			preN_amp = 3
			preN_dur = 2
		elif preN == 'HL5BN1': # Responds real badly to pulses
			preN_amp_step = 0.25 # amplitude of square current step
			preN_dur_step = 150 # duration of square current step
			preN_amp = 3
			preN_dur = 2
		else:
			preN_amp_step = 0.4 # amplitude of square current step
			preN_dur_step = 150 # duration of square current step
			preN_amp = 3
			preN_dur = 2
		
		original_cellnames = [preN, postN]
		origninal_cellnames_copy = original_cellnames.copy()
		
		################################################################################
		# Parameters
		################################################################################
		
		L5_pop_args = {'radius':250,
						'loc':L5depth_shift(L5meandepth),
						'scale':L5rangedepth*4,
						'cap': L5rangedepth}
		
		rotations = {'HL5PN1':{'x':1.57,'y':3.72},
						 'HL5MN1':{'x':1.77,'y':2.77},
						 'HL5BN1':{'x':1.26,'y':2.57},
						 'HL5VN1':{'x':-1.57,'y':3.57}}
		
		OUTPUTPATH = 'Circuit_output/test_syn_results'
		if not os.path.isdir(OUTPUTPATH):
			os.mkdir(OUTPUTPATH)
			print('Created ', OUTPUTPATH)
		
		#somatic potentials
		pre_somav_train = []
		post_somav_train = []
		
		# Currents for currentscapes
		vvec = []
		pIm = []
		pSK = []
		pK_P = []
		pK_T = []
		pKv3_1 = []
		pIl = []
		pIh = []
		pCa_HVA = []
		pCa_LVA = []
		pNap = []
		pNaTg = []
		
		#synapse positions
		post_synlist = []
		syndists = []
		
		#spiketime & cell ID
		spikelist_pre_train = []
		
		h.load_file('net_functions.hoc')
		h.load_file('models/biophys_HL5PN1' + agegroup + '.hoc')
		h.load_file('models/biophys_HL5MN1.hoc')
		h.load_file('models/biophys_HL5BN1.hoc')
		h.load_file('models/biophys_HL5VN1.hoc')
		
		################################################################################
		# Functions
		################################################################################
		
		def generateSubPop(popsize,mname,popargs):
			print('Initiating ' + mname + ' population...')
			morphpath = 'morphologies/' + mname + '.swc'
			templatepath = 'models/NeuronTemplate.hoc'
			templatename = 'NeuronTemplate'
			
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
				'pt3d': True,#use pt3d-info of the cell geometries switch, d=F
				'delete_sections': False,
				'verbose': False}#verbose output switch, for some reason doens't work, figure out why}
			
			rotation = rotations.get(mname)
			
			popParams = {
				'CWD': None,
				'CELLPATH': None,
				'Cell' : LFPy.NetworkCell,
				'POP_SIZE': popsize,
				'name': mname,
				'cell_args' : cellParams,
				'pop_args' : popargs,
				'rotation_args' : rotation}
			
			network.create_population(**popParams)
			
			# Get total cell count from previously initiated populations
			cellcount = 0
			for popname in network.populations:
				if (popname == mname):
					break
				cellcount = cellcount + len(network.populations[popname].cells)
			
			# Add biophys, OU processes, & tonic inhibition to cells
			for cellind in range(0,len(network.populations[mname].cells)): #0 is redundant?
				biophys = 'h.biophys_' + mname + '(network.populations[\'' + mname + '\'].cells[' + str(cellind) + '].template)'
				exec(biophys)
		
		################################################################################
		# Sim
		################################################################################
		new_cellnames = ['to_be_filled', 'to_be_filled']
		for connection in range(N_connection_iterations):
			networkParams = {
				'dt' : dt,
				'tstop' : tstop,
				'v_init' : v_init,
				'celsius' : celsius,
				'OUTPUTPATH' : OUTPUTPATH,
				'verbose' : False
			}
			
			network = Network(**networkParams)
			
			#display run
			print('Running connection '+str(connection+1)+' of '+str(N_connection_iterations)+' connections')
			
			#generate pops
			if original_cellnames[0]!=original_cellnames[1]:
				generateSubPop(1, preN, L5_pop_args)
				generateSubPop(1, postN, L5_pop_args)
				pre1=preN
				post2=postN
			
			if original_cellnames[0]==original_cellnames[1]:
				pre1 = origninal_cellnames_copy[0]
				post1 = origninal_cellnames_copy[1]
				
				#first generate pre3 and post4
				generateSubPop(1, pre1, L5_pop_args)
				pre3 = pre1[:-1]+str(3)
				network.populations[pre3] = network.populations[pre1]
				del network.populations[pre1]
				
				generateSubPop(1, post1, L5_pop_args)
				post4 = post1[:-1]+str(4)
				network.populations[post4] = network.populations[post1]
				del network.populations[post1]
				
				#then rename to pre1, post2
				pre1 = pre3[:-1]+str(1)
				network.populations[pre1] = network.populations[pre3]
				del network.populations[pre3]
				
				post2 = post4[:-1]+str(2)
				network.populations[post2] = network.populations[post4]
				del network.populations[post4]
			
			new_cellnames[0]=pre1
			new_cellnames[1]=post2
			
			# Setup for currentscapes
			tempCell = network.populations[new_cellnames[1]].cells[0].template
			t_vec = h.Vector()
			t_vec.record(h._ref_t)
			v_vec = h.Vector()
			v_vec.record(tempCell.soma[0](0.5)._ref_v)
			Ca_HVA = h.Vector()
			Ca_HVA.record(tempCell.soma[0](0.5)._ref_ica_Ca_HVA)
			Ca_LVA = h.Vector()
			Ca_LVA.record(tempCell.soma[0](0.5)._ref_ica_Ca_LVA)
			Ih = h.Vector()
			Ih.record(tempCell.soma[0](0.5)._ref_ihcn_Ih)
			Im = h.Vector()
			Im.record(tempCell.soma[0](0.5)._ref_ik_Im)
			K_P = h.Vector()
			K_P.record(tempCell.soma[0](0.5)._ref_ik_K_P)
			K_T = h.Vector()
			K_T.record(tempCell.soma[0](0.5)._ref_ik_K_T)
			Kv3_1 = h.Vector()
			Kv3_1.record(tempCell.soma[0](0.5)._ref_ik_Kv3_1)
			Nap = h.Vector()
			Nap.record(tempCell.soma[0](0.5)._ref_ina_Nap)
			NaTg = h.Vector()
			NaTg.record(tempCell.soma[0](0.5)._ref_ina_NaTg)
			SK = h.Vector()
			SK.record(tempCell.soma[0](0.5)._ref_ik_SK)
			Il = h.Vector()
			Il.record(tempCell.soma[0](0.5)._ref_i_pas)
			
			connectionProbability = np.array([[0, 1],[0, 0]]) #only Pre->Post connection
			
			E_syn = neuron.h.ProbAMPANMDA
			I_syn = neuron.h.ProbUDFsyn
			
			weightFunction = local_state.normal
			weightParams = {'loc':1, 'scale':0}
			minweight=1
			
			delayFunction = local_state.normal
			delayParams = {'loc':.5, 'scale':0}
			mindelay=.5
			
			multapseFunction = local_state.normal
			
			prec = new_cellnames[0]
			postc = new_cellnames[1]
			
			synapseParameters = [[syn_params['none'],syn_params[prec+postN]],[syn_params['none'],syn_params['none']]]
			weightArguments = [[weightParams, weightParams],[weightParams, weightParams]]
			delayArguments = [[delayParams, delayParams],[delayParams, delayParams]]
			multapseArguments = [[mult_syns['none'],mult_syns[prec+postN]],[mult_syns['none'], mult_syns['none']]]
			synapsePositionArguments = [[pos_args['none'],pos_args[prec+postN]],[pos_args['none'],pos_args['none']]]
			
			for i, prec in enumerate(new_cellnames):
				for j, postc in enumerate(new_cellnames):
					connectivity = network.get_connectivity_rand(pre=prec,post=postc,connprob=connectionProbability[i][j])
					
					(conncount, syncount) = network.connect(
						pre=prec, post=postc,
						connectivity=connectivity,
						syntype=E_syn if prec=='HL5PN1' else I_syn,
						synparams=synapseParameters[i][j],
						weightfun=weightFunction,
						weightargs=weightArguments[i][j],
						minweight=minweight,
						delayfun=delayFunction,
						delayargs=delayArguments[i][j],
						mindelay=mindelay,
						multapsefun=multapseFunction,
						multapseargs=multapseArguments[i][j],
						syn_pos_args=synapsePositionArguments[i][j])
			
			# Record the sampled synapse distances and set the random seed for each synapses
			h.distance(sec=network.populations[postc].cells[0].template.soma[0])
			h.pop_section()
			for l in range(0,len(network.populations[postc].cells[0].netconsynapses)):
				locx = network.populations[postc].cells[0].netconsynapses[l].get_loc()
				syndists.append(h.distance(locx))
				
				# Replicates LFPy's method but with control over seeds
				rseed = local_state.randint(0,2**32 - 1)
				rng = neuron.h.Random(rseed)
				rng.MCellRan4(
					local_state.randint(
						0,
						2**32 - 1),
					local_state.randint(
						0,
						2**32 - 1))
				rng.uniform(0, 1)
				network.populations[postc].cells[0].netconsynapses[l].setRNG(rng)
				network.populations[postc].cells[0].rng_list[l] = rng
				
				h.pop_section()
			
			# Setup protocols
			for name, pop in network.populations.items():
				if name==new_cellnames[0]:
					for cell in pop.cells:
						delays = train_delay
						pointprocesses = []
						pre_stimuli = []
						for i in range(len(delays)):
							if step_instead_of_train:
								pointprocesses.append({
									'idx' : 0,
									'record_current' : True,
									'pptype' : 'IClamp',
									'amp' : preN_amp_step,
									'dur' : preN_dur_step,
									'delay': delays[0]})
								break
							else:
								pointprocesses.append({
									'idx' : 0,
									'record_current' : True,
									'pptype' : 'IClamp',
									'amp' : preN_amp,
									'dur' : preN_dur,
									'delay': delays[i]})
						for pointprocess in pointprocesses:pre_stimuli.append(LFPy.StimIntElectrode(cell, **pointprocess))
				if name==new_cellnames[1]:
					for cell in pop.cells:
						pointprocess = {
							'idx' : 0,
							'record_current' : True,
							'pptype' : 'IClamp',
							'amp' : postN_amp,
							'dur' : postN_dur,
							'delay': 0}
						post_stimuli = LFPy.StimIntElectrode(cell, **pointprocess)
			
			simargs = {'electrode': None,
					   'rec_imem': False,
					   'rec_vmem': False,
					   'rec_ipas': False,
					   'rec_icap': False,
					   'rec_isyn': False,
					   'rec_vmemsyn': False,
					   'rec_istim': False}
			
			print('Stimulating '+str(new_cellnames[0])+'...')
			
			pst = []
			pvvec = []
			ppIm = []
			ppSK = []
			ppK_P = []
			ppK_T = []
			ppKv3_1 = []
			ppIl = []
			ppIh = []
			ppCa_HVA = []
			ppCa_LVA = []
			ppNap = []
			ppNaTg = []
			
			# Run N trials for each connection
			for trial in range(N_trial_iterations):
				SPIKES = network.simulate(**simargs)
				
				#save somatic potentials across trials
				for name, pop in network.populations.items():
					if name==new_cellnames[0]:
						for cell in pop.cells:
							pre_somav_train.append(cell.somav)
							spikelist_pre_train.append(SPIKES['times'][0])
					if name==new_cellnames[1]:
						for cell in pop.cells:
							pst.append(np.array(cell.somav))
							# Voltage
							pvvec.append(np.array(v_vec))
							# Outward
							ppIm.append(np.array(Im))
							ppSK.append(np.array(SK))
							ppK_P.append(np.array(K_P))
							ppK_T.append(np.array(K_T))
							ppKv3_1.append(np.array(Kv3_1))
							ppIl.append(np.array(Il))
							# Inward
							ppIh.append(np.array(Ih))
							ppCa_HVA.append(np.array(Ca_HVA))
							ppCa_LVA.append(np.array(Ca_LVA))
							ppNap.append(np.array(Nap))
							ppNaTg.append(np.array(NaTg))
							
							post_synlist.append(cell.synidx)
			
			if bootCI_trials:
				bspst = []
				for l in range(0,len(pst[0])):
					x = bs.bootstrap(np.transpose(pst)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
					bspst.append(x.value)
				post_somav_train.append(bspst)
			else:
				post_somav_train.append(np.mean(pst,axis=0))
			
			# Voltage
			vvec.append(np.mean(pvvec,axis=0))
			# Outward
			pIm.append(np.mean(ppIm,axis=0))
			pSK.append(np.mean(ppSK,axis=0))
			pK_P.append(np.mean(ppK_P,axis=0))
			pK_T.append(np.mean(ppK_T,axis=0))
			pKv3_1.append(np.mean(ppKv3_1,axis=0))
			pIl.append(np.mean(ppIl,axis=0))
			# Inward
			pIh.append(np.mean(ppIh,axis=0))
			pCa_HVA.append(np.mean(ppCa_HVA,axis=0))
			pCa_LVA.append(np.mean(ppCa_LVA,axis=0))
			pNap.append(np.mean(ppNap,axis=0))
			pNaTg.append(np.mean(ppNaTg,axis=0))
			
			#clear to allow for iteration, don't clear last so we can grab objects from script and use in plot
			if connection < N_connection_iterations-1:
				network.pc.gid_clear()
				electrode = None
				syn = None
				synapseModel = None
				for population in network.populations.values():
					for cell in population.cells:
						cell = None
						population.cells = None
					population = None
					pop = None
					network = None
					neuron.h('forall delete_section()')
			
			if connection==N_connection_iterations-1:
				
				np.save(OUTPUTPATH+'/spikelist_pre_train.npy', spikelist_pre_train, allow_pickle=True)
				
				t1 = int(startslice/network.dt)
				t2 = int(endslice/network.dt)
				
				t3 = int(tmid-50)
				t4 = int(tmid-1)
				t1_rest = int(t3/network.dt)
				t2_rest = int(t4/network.dt)
				
				np.save(OUTPUTPATH+'/pre_somav_train_'+preN+postN+'_'+agegroup, pre_somav_train)
				np.save(OUTPUTPATH+'/post_somav_train_'+preN+postN+'_'+agegroup, post_somav_train)
				
				font = {'family' : 'DejaVu Sans',
						'size'   : 25}
				matplotlib.rc('font', **font)
				tvec = np.arange(network.tstop / network.dt + 1) * network.dt
				
				#Post Morph
				if PlotMorphs:
					ax[pidx].axis('off')
					for name, pop in network.populations.items():
						if preN == 'HL5PN1': col3 = 'b'
						if preN == 'HL5MN1': col3 = 'crimson'
						if preN == 'HL5BN1': col3 = 'g'
						if preN == 'HL5VN1': col3 = 'orange'
						if postN == 'HL5PN1': col4 = 'b'
						if postN == 'HL5MN1': col4 = 'crimson'
						if postN == 'HL5BN1': col4 = 'g'
						if postN == 'HL5VN1': col4 = 'orange'
						if ((name==new_cellnames[1]) & (agegroup=='o')):
							for cell in pop.cells:
								for i, idx in enumerate(cell.synidx):
									ax[pidx].plot(cell.ymid[idx], cell.zmid[idx], c=col3, marker='.', markersize='15')
								for i, idx in enumerate(np.concatenate(post_synlist).tolist()):
									ax[pidx].plot(cell.ymid[idx], cell.zmid[idx], c=col3, marker='.', markersize='15', alpha=0.1)
								zips = []
								for x, z in cell.get_pt3d_polygons(projection=('y', 'z')):
									zips.append(list(zip(x, z)))
									polycol = PolyCollection(zips,
														edgecolors='none',
														facecolors=col4)
									ax[pidx].add_collection(polycol)
									# ax[1][pidx].axis('equal')
									ax[pidx].set_xticks([])
									ax[pidx].set_yticks([])
						if ((name==new_cellnames[0]) & (agegroup=='o')):
							for cell in pop.cells:
								zips = []
								for x, z in cell.get_pt3d_polygons(projection=('y', 'z')):
									zips.append(list(zip(x, z)))
									polycol = PolyCollection(zips,
														edgecolors='none',
														facecolors=col3)
									ax[pidx].add_collection(polycol)
									# ax[1][pidx].axis('equal')
									ax[pidx].set_xticks([])
									ax[pidx].set_yticks([])
				
				ax2[pidx].axis('off')
				
				# Compute and plot mean PSPs across connections
				post_somav_train = [np.array(p) for p in post_somav_train]
				if bootCI_seeds:
					post_somav_m_train = []
					post_somav_l_train = []
					post_somav_u_train = []
					for l in range(0,len(post_somav_train[0])):
						x = bs.bootstrap(np.transpose(post_somav_train)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
						post_somav_m_train.append(x.value)
						post_somav_l_train.append(x.lower_bound)
						post_somav_u_train.append(x.upper_bound)
				else:
					post_somav_m_train = np.mean(post_somav_train,axis=0)
					post_somav_sd_train = np.std(post_somav_train,axis=0)
				post_somav_m_RMP = np.mean(post_somav_m_train[t1_rest:t2_rest])
				
				post_somav_areas[pidx][aidx] = [np.trapz(np.abs(p[t1:t2]-post_somav_m_RMP),x=tvec[t1:t2]) for p in post_somav_train]
				post_somav_amplitude[pidx][aidx] = [np.max(np.abs(p[t1:t2]-post_somav_m_RMP)) for p in post_somav_train]
				post_somav_mconnection_traces[pidx][aidx] = [p[t1:t2]-post_somav_m_RMP for p in post_somav_train]
				
				if not compensate:
					if agegroup == 'y':
						if preN == 'HL5PN1': RMPy_PN = post_somav_m_RMP
						if preN == 'HL5MN1': RMPy_MN = post_somav_m_RMP
						if preN == 'HL5BN1': RMPy_BN = post_somav_m_RMP
					elif ((agegroup == 'o') & (shifttrace)):
						if preN == 'HL5PN1': post_somav_m_train = post_somav_m_train - abs(post_somav_m_RMP - RMPy_PN)
						if preN == 'HL5MN1': post_somav_m_train = post_somav_m_train - abs(post_somav_m_RMP - RMPy_MN)
						if preN == 'HL5BN1': post_somav_m_train = post_somav_m_train - abs(post_somav_m_RMP - RMPy_BN)
				
				if bootCI_seeds:
					ax2[pidx].fill_between(tvec[t1:t2], post_somav_u_train[t1:t2], post_somav_l_train[t1:t2], facecolor=col2, alpha=0.5)
				else:
					ax2[pidx].fill_between(tvec[t1:t2], post_somav_m_train[t1:t2]+post_somav_sd_train[t1:t2], post_somav_m_train[t1:t2]-post_somav_sd_train[t1:t2], facecolor=col2, alpha=0.5)
				ax2[pidx].plot(tvec[t1:t2], post_somav_m_train[t1:t2], c=col1,linewidth=4)
				ax2[pidx].set_xticks([])
				ax2[pidx].set_xlim(tvec[t1],tvec[t2])
				
				if preN != 'HL5PN1':
					if preN == 'HL5MN1': colc = 'crimson'
					if preN == 'HL5BN1': colc = 'g'
					tbase = int(int(tmid-1)/network.dt)
					tpeak = int(int(tmid+10)/network.dt)
					if preN == 'HL5MN1': scale_v = abs(post_somav_m_train[tpeak]-post_somav_m_train[tbase])
					if bootCI_seeds:
						ax3[aidx].fill_between(tvec[t1:t2], post_somav_u_train[t1:t2], post_somav_l_train[t1:t2], facecolor=colc, alpha=0.5)
					else:
						ax3[aidx].fill_between(tvec[t1:t2], post_somav_m_train[t1:t2]+post_somav_sd_train[t1:t2], post_somav_m_train[t1:t2]-post_somav_sd_train[t1:t2], facecolor=colc, alpha=0.5)
					ax3[aidx].plot(tvec[t1:t2], post_somav_m_train[t1:t2], c=colc,linewidth=4)
					ax3[aidx].set_xticks([])
					ax3[aidx].set_xlim(tvec[t1],tvec[t2])
					ax3[aidx].axis('off')
					if preN != 'HL5BN1':
						xmin = tvec[t2]-75
						xmax = tvec[t2]-25
						ymin = post_somav_m_train[t2]-1
						ymax = post_somav_m_train[t2]-0.5
						xlength = xmax-xmin
						ylength = ymax-ymin
						ax3[aidx].plot([xmin,xmax],[ymin,ymin],'k',linewidth=4)
						ax3[aidx].plot([xmax,xmax],[ymin,ymax],'k',linewidth=4)
						ax3[aidx].text(xmax-xlength-20,ymin-0.25, str(xlength) +' ms')
						ax3[aidx].text(xmax+3,ymax-ylength, str(ylength) +' mV',rotation=90)
				
				if ((agegroup == 'o') & (shifttrace)):
					xmin = tvec[t2]-75
					xmax = tvec[t2]-25
					if preN == 'HL5PN1':
						ymin = post_somav_m_train[t2]+0.5
						ymax = post_somav_m_train[t2]+1
					else:
						ymin = post_somav_m_train[t2]-1
						ymax = post_somav_m_train[t2]-0.5
					xlength = xmax-xmin
					ylength = ymax-ymin
					ax2[pidx].plot([xmin,xmax],[ymin,ymin],'k',linewidth=4)
					ax2[pidx].plot([xmax,xmax],[ymin,ymax],'k',linewidth=4)
					ax2[pidx].text(xmax-xlength-20,ymin-0.25, str(xlength) +' ms')
					ax2[pidx].text(xmax+3,ymax-ylength, str(ylength) +' mV',rotation=90)
				else:
					if ((preN == 'HL5PN1') & (agegroup == 'o')):
						xmin = tvec[t2]-75
						xmax = tvec[t2]-25
						ymin = post_somav_m_train[t2]+1
						ymax = post_somav_m_train[t2]+1.5
						xlength = int(xmax-xmin)
						ylength = ymax-ymin
						ax2[pidx].plot([xmin,xmax],[ymin,ymin],'k',linewidth=4)
						ax2[pidx].plot([xmax,xmax],[ymin,ymax],'k',linewidth=4)
						if compensate:
							ax2[pidx].text(xmax-xlength-5,ymin-0.30, str(xlength) +' ms')
							ax2[pidx].text(xmax+3,ymax-ylength-0.25, str(ylength) +' mV',rotation=90)
						else:
							ax2[pidx].text(xmax-xlength-5,ymin-0.60, str(xlength) +' ms')
							ax2[pidx].text(xmax+3,ymax-ylength-0.25, str(ylength) +' mV',rotation=90)
					elif ((preN != 'HL5PN1') & (agegroup == 'y')):
						xmin = tvec[t2]-75
						xmax = tvec[t2]-25
						if preN == 'HL5MN1':
							ymin = post_somav_m_train[t2]-1.5
							ymax = post_somav_m_train[t2]-1
							xlength = int(xmax-xmin)
							ylength = ymax-ymin
							ax2[pidx].plot([xmin,xmax],[ymin,ymin],'k',linewidth=4)
							ax2[pidx].plot([xmax,xmax],[ymin,ymax],'k',linewidth=4)
							if compensate:
								ax2[pidx].text(xmax-xlength-5,ymin-0.40, str(xlength) +' ms')
								ax2[pidx].text(xmax+3,ymax-ylength-0.25, str(ylength) +' mV',rotation=90)
							else:
								ax2[pidx].text(xmax-xlength-5,ymin-0.50, str(xlength) +' ms')
								ax2[pidx].text(xmax+3,ymax-ylength-0.5, str(ylength) +' mV',rotation=90)
						elif preN == 'HL5BN1':
							ymin = post_somav_m_train[t2]-2
							ymax = post_somav_m_train[t2]-1
							xlength = int(xmax-xmin)
							ylength = ymax-ymin
							ax2[pidx].plot([xmin,xmax],[ymin,ymin],'k',linewidth=4)
							ax2[pidx].plot([xmax,xmax],[ymin,ymax],'k',linewidth=4)
							if compensate:
								ax2[pidx].text(xmax-xlength-5,ymin-0.60, str(xlength) +' ms')
								ax2[pidx].text(xmax+3,ymax-ylength-0.25, str(ylength) +' mV',rotation=90)
							else:
								ax2[pidx].text(xmax-xlength-5,ymin-0.65, str(xlength) +' ms')
								ax2[pidx].text(xmax+3,ymax-ylength-0.45, str(ylength) +' mV',rotation=90)
				
				TStart = t1
				TEnd = t2
				vvec_m = np.mean(vvec,axis=0)
				# Outward
				pIm_m = np.mean(pIm,axis=0)
				pSK_m = np.mean(pSK,axis=0)
				pK_P_m = np.mean(pK_P,axis=0)
				pK_T_m = np.mean(pK_T,axis=0)
				pKv3_1_m = np.mean(pKv3_1,axis=0)
				pIl_m = np.mean(pIl,axis=0)
				# Inward
				pIh_m = np.mean(pIh,axis=0)
				pCa_HVA_m = np.mean(pCa_HVA,axis=0)
				pCa_LVA_m = np.mean(pCa_LVA,axis=0)
				pNap_m = np.mean(pNap,axis=0)
				pNaTg_m = np.mean(pNaTg,axis=0)
				r0 = [pIm_m[TStart:TEnd],pSK_m[TStart:TEnd],pK_P_m[TStart:TEnd],pK_T_m[TStart:TEnd],pKv3_1_m[TStart:TEnd],pIl_m[TStart:TEnd],pIh_m[TStart:TEnd],pCa_HVA_m[TStart:TEnd],pCa_LVA_m[TStart:TEnd],pNap_m[TStart:TEnd],pNaTg_m[TStart:TEnd]]
				labels = ('$I_{M}$','$I_{SK}$','$I_{K_P}$','$I_{K_T}$','$I_{Kv3.1}$','$I_L$','$I_{H}$','$I_{Ca_{HVA}}$','$I_{Ca_{LVA}}$','$I_{Nap}$','$I_{NaTg}$')
				
				fig0 = plotCurrentscape(vvec_m[TStart:TEnd], r0)
				filename = 'Currentscape_' + str(preN) + '_'+str(agegroup)+'.png'
				fig0.savefig(os.path.join(OUTPUTPATH,filename),dpi=500,transparent=True)
				fig0.clf()
				plt.close(fig0)
				
				Set3Colors = plt.cm.Set3(np.linspace(0, 1, len(r0)))
				f, axarr = plt.subplots(6, 2, sharex=True, figsize=(15,10))
				axarr[0,0].plot(tvec[TStart:TEnd],vvec_m[TStart:TEnd],color='k',linestyle='solid',linewidth=5)
				axarr[0,0].set_ylabel('$V_{m}$')
				axarr[1,0].plot(tvec[TStart:TEnd],r0[0]*1000,color=Set3Colors[0],linestyle='solid',linewidth=5)
				axarr[1,0].set_ylabel(labels[0])
				axarr[2,0].plot(tvec[TStart:TEnd],r0[1]*1000,color=Set3Colors[1],linestyle='solid',linewidth=5)
				axarr[2,0].set_ylabel(labels[1])
				axarr[3,0].plot(tvec[TStart:TEnd],r0[2]*1000,color=Set3Colors[2],linestyle='solid',linewidth=5)
				axarr[3,0].set_ylabel(labels[2])
				axarr[4,0].plot(tvec[TStart:TEnd],r0[3]*1000,color=Set3Colors[3],linestyle='solid',linewidth=5)
				axarr[4,0].set_ylabel(labels[3])
				axarr[5,0].plot(tvec[TStart:TEnd],r0[4]*1000,color=Set3Colors[4],linestyle='solid',linewidth=5)
				axarr[5,0].set_ylabel(labels[4])
				axarr[0,1].plot(tvec[TStart:TEnd],r0[5]*1000,color=Set3Colors[5],linestyle='solid',linewidth=5)
				axarr[0,1].set_ylabel(labels[5])
				axarr[1,1].plot(tvec[TStart:TEnd],r0[6]*1000,color=Set3Colors[6],linestyle='solid',linewidth=5)
				axarr[1,1].set_ylabel(labels[6])
				axarr[2,1].plot(tvec[TStart:TEnd],r0[7]*1000,color=Set3Colors[7],linestyle='solid',linewidth=5)
				axarr[2,1].set_ylabel(labels[7])
				axarr[3,1].plot(tvec[TStart:TEnd],r0[8]*1000,color=Set3Colors[8],linestyle='solid',linewidth=5)
				axarr[3,1].set_ylabel(labels[8])
				axarr[4,1].plot(tvec[TStart:TEnd],r0[9]*1000,color=Set3Colors[9],linestyle='solid',linewidth=5)
				axarr[4,1].set_ylabel(labels[9])
				axarr[5,1].plot(tvec[TStart:TEnd],r0[10]*1000,color=Set3Colors[10],linestyle='solid',linewidth=5)
				axarr[5,1].set_ylabel(labels[10])
				axarr[0,0].set_xlim(tvec[TStart],tvec[TEnd])
				
				for o in range(0,len(axarr)):
					for u in range(0,len(axarr[o])):
						axarr[o,u].ticklabel_format(axis='y', style='sci',scilimits=[-2, 2])
				f.subplots_adjust(hspace=0.5, wspace=0.4)
				filename = 'AllCurrents_' + str(preN) + '_'+str(agegroup)+'.png'
				f.savefig(os.path.join(OUTPUTPATH,filename),dpi=500,transparent=True)
				f.clf()
				plt.close(f)
				
				network.pc.gid_clear()
		print('Simulation complete')

fig.savefig(os.path.join(OUTPUTPATH,'SynapseLocations'),bbox_inches='tight',dpi=300,transparent=True)
fig2.savefig(os.path.join(OUTPUTPATH,'YoungVsOld_Integration'),bbox_inches='tight',dpi=300,transparent=True)
fig3.savefig(os.path.join(OUTPUTPATH,'YoungVsOld_Integration_MNvsBN'),bbox_inches='tight',dpi=300,transparent=True)
plt.close()

# Old/Young PSP Differences
np.save(OUTPUTPATH+'/post_somav_all_absRMP_equalized_traces', post_somav_mconnection_traces)
for i in range(0,len(post_somav_mconnection_traces)):
	young = post_somav_mconnection_traces[i][0]
	old = post_somav_mconnection_traces[i][1]
	if preNs[i] == 'HL5PN1':
		diffs = [y-o for y,o in zip(young,old)]
	else:
		diffs = [-y--o for y,o in zip(young,old)]
	diffs_m = []
	diffs_l = []
	diffs_u = []
	for l in range(0,len(diffs[0])):
		x = bs.bootstrap(np.transpose(diffs)[l], stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		diffs_m.append(x.value)
		diffs_l.append(x.lower_bound)
		diffs_u.append(x.upper_bound)
	
	fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5),sharex=True,sharey=False,frameon=False)
	ax.fill_between(tvec[t1:t2], diffs_u, diffs_l, facecolor='dimgray', alpha=0.5)
	ax.plot(tvec[t1:t2], diffs_m, c='k',linewidth=4)
	ax.set_xlim(tvec[t1],tvec[t2])
	ax.set_ylabel(r'$|PSP_{Y}|-|PSP_{O}|$')
	ax.set_xlabel('Time (ms)')
	plt.tight_layout()
	fig.savefig(os.path.join(OUTPUTPATH,'PSPDiffs_'+preNs[i]+'.png'),bbox_inches='tight',dpi=300,transparent=True)
	plt.close()

# Stats PIDX = PN,MN,BN; AID = Y,O
def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

df = pd.DataFrame(columns=["Connection",
			"Metric",
			"Mean Young",
			"SD Young",
			"Mean Old",
			"SD Old",
			"Normal Test Young",
			"Normal Test Old",
			"t-test stat",
			"t-test p-value",
			"MWU stat",
			"MWU p-value",
			"Cohen's d"])

for i in range(0,len(post_somav_amplitude)):
	m_y2 = np.mean(post_somav_amplitude[i][0])
	m_o2 = np.mean(post_somav_amplitude[i][1])
	sd_y2 = np.std(post_somav_amplitude[i][0])
	sd_o2 = np.std(post_somav_amplitude[i][1])
	n_stat_y2, n_pval_y2 = st.normaltest(post_somav_amplitude[i][0])
	n_stat_o2, n_pval_o2 = st.normaltest(post_somav_amplitude[i][1])
	tstat2, pval2 = st.ttest_rel(post_somav_amplitude[i][0],post_somav_amplitude[i][1])
	mwu_stat2, mwu_pval2 = st.mannwhitneyu(post_somav_amplitude[i][0],post_somav_amplitude[i][1])
	cd2 = cohen_d(post_somav_amplitude[i][0],post_somav_amplitude[i][1])
	
	df = df.append({"Connection" : preNs[i]+postN,
				"Metric" : 'PSP Amplitude',
				"Mean Young" : m_y2,
				"SD Young" : sd_y2,
				"Mean Old" : m_o2,
				"SD Old" : sd_o2,
				"Normal Test Young" : n_pval_y2,
				"Normal Test Old" : n_pval_o2,
				"t-test stat" : tstat2,
				"t-test p-value" : pval2,
				"MWU stat" : mwu_stat2,
				"MWU p-value" : mwu_pval2,
				"Cohen's d" : cd2},
				ignore_index = True)
	
	x = [-0.2, 1.2]
	c1 = 'dimgray'
	c2 = 'lightcoral'
	fig, ax = plt.subplots(figsize=(4, 5))
	ax.boxplot(post_somav_amplitude[i][0],
						positions=[x[0]],
						widths=1,
						notch=True,
						patch_artist=True,
						boxprops=dict(facecolor=c1, color=c1),
						capprops=dict(color=c1),
						whiskerprops=dict(color=c1),
						flierprops=dict(color=c1, markeredgecolor=c1),
						medianprops=dict(color=c1))
	ax.boxplot(post_somav_amplitude[i][1],
						positions=[x[1]],
						widths=1,
						notch=True,
						patch_artist=True,
						boxprops=dict(facecolor=c2, color=c2),
						capprops=dict(color=c2),
						whiskerprops=dict(color=c2),
						flierprops=dict(color=c2, markeredgecolor=c2),
						medianprops=dict(color=c2))
	ax.set_xlim(-1,2)
	ax.set_xticks(x)
	ax.set_xticklabels(['Younger','Older'])
	ax.set_ylabel('PSP Amplitude (mV)')
	ax.grid(False)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig(os.path.join(OUTPUTPATH,'PSPAmps_boxplots_'+preNs[i]+'.png'),bbox_inches='tight',dpi=300,transparent=True)
	plt.close()

for i in range(0,len(post_somav_areas)):
	m_y = np.mean(post_somav_areas[i][0])
	m_o = np.mean(post_somav_areas[i][1])
	sd_y = np.std(post_somav_areas[i][0])
	sd_o = np.std(post_somav_areas[i][1])
	n_stat_y, n_pval_y = st.normaltest(post_somav_areas[i][0])
	n_stat_o, n_pval_o = st.normaltest(post_somav_areas[i][1])
	tstat, pval = st.ttest_rel(post_somav_areas[i][0],post_somav_areas[i][1])
	mwu_stat, mwu_pval = st.mannwhitneyu(post_somav_areas[i][0],post_somav_areas[i][1])
	cd = cohen_d(post_somav_areas[i][0],post_somav_areas[i][1])
	
	df = df.append({"Connection" : preNs[i]+postN,
				"Metric" : 'Area Under PSP',
				"Mean Young" : m_y,
				"SD Young" : sd_y,
				"Mean Old" : m_o,
				"SD Old" : sd_o,
				"Normal Test Young" : n_pval_y,
				"Normal Test Old" : n_pval_o,
				"t-test stat" : tstat,
				"t-test p-value" : pval,
				"MWU stat" : mwu_stat,
				"MWU p-value" : mwu_pval,
				"Cohen's d" : cd},
				ignore_index = True)
	
	x = [-0.2, 1.2]
	c1 = 'dimgray'
	c2 = 'lightcoral'
	fig, ax = plt.subplots(figsize=(4, 5))
	ax.boxplot(post_somav_areas[i][0],
						positions=[x[0]],
						widths=1,
						notch=True,
						patch_artist=True,
						boxprops=dict(facecolor=c1, color=c1),
						capprops=dict(color=c1),
						whiskerprops=dict(color=c1),
						flierprops=dict(color=c1, markeredgecolor=c1),
						medianprops=dict(color=c1))
	ax.boxplot(post_somav_areas[i][1],
						positions=[x[1]],
						widths=1,
						notch=True,
						patch_artist=True,
						boxprops=dict(facecolor=c2, color=c2),
						capprops=dict(color=c2),
						whiskerprops=dict(color=c2),
						flierprops=dict(color=c2, markeredgecolor=c2),
						medianprops=dict(color=c2))
	ax.set_xlim(-1,2)
	ax.set_xticks(x)
	ax.set_xticklabels(['Younger','Older'])
	ax.set_ylabel('Area Under PSP')
	ax.grid(False)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig(os.path.join(OUTPUTPATH,'PSPAUC_boxplots_'+preNs[i]+'.png'),bbox_inches='tight',dpi=300,transparent=True)
	plt.close()

df.to_csv(os.path.join(OUTPUTPATH,'stats_PSPs.csv'))
