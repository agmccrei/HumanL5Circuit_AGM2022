################################
### Setup Recording Protocol ###
################################
h.tstop = 2000
h.dt = 0.025
stepnames = ['step'+str(l) for l in numpy.linspace(1,len(c_soma),len(c_soma),dtype=numpy.int)]
zipsteps = zip(stepnames,c_soma)

######### Choose Recording Sites #########
# Important: Do not use the re-assign the loc variables initialized in init_1morphology.py
soma_loc = ephys.locations.NrnSeclistCompLocation(
	name=rec1name,
	seclist_name='somatic',
	sec_index = 0,
	comp_x = 0.5
)

axon_loc = ephys.locations.NrnSeclistCompLocation(
	name=rec2name,
	seclist_name='axonal',
	sec_index = 0,
	comp_x = 0.5
)
nrn = ephys.simulators.NrnSimulator()

######### Create Stimuli for Active Optimizations #########
sweep_protocols = []
stim_protocol = []
rec_protocol = []
for protocol_name, amplitude in zipsteps:
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
	sweep_protocols.append(protocol)

recording_protocol = ephys.protocols.SequenceProtocol(name=str(len(stepnames))+'step', protocols=sweep_protocols)
