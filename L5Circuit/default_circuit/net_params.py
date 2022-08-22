# Ornstein-Uhlenbeck Excitatory Conductances
Gou_PN = 0.000076
Gou_MN = 0.000032
Gou_BN = 0.001545
Gou_VN = 0.000075

# Tonic Inhibition Conductances
ADDdrug = False
if ADDdrug:
	Gtonic_PN = 0.001352
	Gtonic_MN = 0.001030
	Gtonic_BN = 0.001091
	Gtonic_VN = 0.000938
	
	GtonicApic_PN = Gtonic_PN
else:
	Gtonic_PN = 0.000938
	Gtonic_MN = 0.000938
	Gtonic_BN = 0.000938
	Gtonic_VN = 0.000938
	
	GtonicApic_PN = Gtonic_PN

# Connection Probability
connection_prob = {
	'HL5PN1HL5PN1' : 0.09,
	'HL5PN1HL5MN1' : 0.09,
	'HL5PN1HL5BN1' : 0.09,
	'HL5PN1HL5VN1' : 0.08,
	
	'HL5MN1HL5PN1' : 0.10,
	'HL5MN1HL5MN1' : 0.09,
	'HL5MN1HL5BN1' : 0.10,
	'HL5MN1HL5VN1' : 0.10,
	
	'HL5BN1HL5PN1' : 0.06,
	'HL5BN1HL5MN1' : 0.16,
	'HL5BN1HL5BN1' : 0.40,
	'HL5BN1HL5VN1' : 0.10,
	
	'HL5VN1HL5PN1' : 0.00,
	'HL5VN1HL5MN1' : 0.30,
	'HL5VN1HL5BN1' : 0.20,
	'HL5VN1HL5VN1' : 0.05}

#Synaptic Conductances
con_PN2PN = 0.00037 # BBP = 0.00031 // Seeman et al (2018) = 0.00037
con_PN2MN = 0.0003
con_PN2BN = 0.0003
con_PN2VN = 0.0003

con_MN2PN = 0.00134 # BBP = 0.00066
con_MN2MN = 0.00033
con_MN2BN = 0.00034
con_MN2VN = 0.00031

con_BN2PN = 0.00092 # BBP = 0.00092
con_BN2MN = 0.00033
con_BN2BN = 0.00034
con_BN2VN = 0.00031 # Same as SBC-DBC since no LBC-BP/LBC-DBC/SBC-BP connections in BBP

con_VN2PN = 0 # Changed to 0 since assumed to be interneuron specific
con_VN2MN = 0.00033
con_VN2BN = 0.00031
con_VN2VN = 0.0004 # Same as DBC-DBC since no BP-BP connection in BBP

#Numbers of synaptic contacts per connection
Ncont_PN2PN = 6
Ncont_PN2MN = 9
Ncont_PN2BN = 8
Ncont_PN2VN = 4

Ncont_MN2PN = 14
Ncont_MN2MN = 12
Ncont_MN2BN = 13
Ncont_MN2VN = 5

Ncont_BN2PN = 22
Ncont_BN2MN = 15
Ncont_BN2BN = 15
Ncont_BN2VN = 12 # Same as SBC-DBC since no LBC-BP/LBC-DBC/SBC-BP connections in BBP

Ncont_VN2PN = 0 # Changed to 0 since assumed to be interneuron specific
Ncont_VN2MN = 12
Ncont_VN2BN = 15
Ncont_VN2VN = 7 # Same as DBC-DBC since no BP-BP connection in BBP

# Rise time
taur_PN2PN = 0.3
taur_PN2MN = 0.3
taur_PN2BN = 0.3
taur_PN2VN = 0.3

taur_MN2PN = 1
taur_MN2MN = 1
taur_MN2BN = 1
taur_MN2VN = 1

taur_BN2PN = 1
taur_BN2MN = 1
taur_BN2BN = 1
taur_BN2VN = 1

taur_VN2PN = 1
taur_VN2MN = 1
taur_VN2BN = 1
taur_VN2VN = 1

# Decay time
taud_PN2PN = 3
taud_PN2MN = 3
taud_PN2BN = 3
taud_PN2VN = 3

taud_MN2PN = 10
taud_MN2MN = 10
taud_MN2BN = 10
taud_MN2VN = 10

taud_BN2PN = 10
taud_BN2MN = 10
taud_BN2BN = 10
taud_BN2VN = 10

taud_VN2PN = 10
taud_VN2MN = 10
taud_VN2BN = 10
taud_VN2VN = 10

# Depression
d_PN2PN = 670
d_PN2MN = 150
d_PN2BN = 580
d_PN2VN = 670

d_MN2PN = 1200
d_MN2MN = 710
d_MN2BN = 710
d_MN2VN = 670

d_BN2PN = 660
d_BN2MN = 700
d_BN2BN = 720
d_BN2VN = 680 # Same as SBC-DBC since no LBC-BP/LBC-DBC/SBC-BP connections in BBP

d_VN2PN = 360
d_VN2MN = 700
d_VN2BN = 680
d_VN2VN = 810 # Same as DBC-DBC since no BP-BP connection in BBP

# Facilitation
f_PN2PN = 17
f_PN2MN = 690
f_PN2BN = 120
f_PN2VN = 17

f_MN2PN = 2.2
f_MN2MN = 21
f_MN2BN = 21
f_MN2VN = 20

f_BN2PN = 27
f_BN2MN = 21
f_BN2BN = 21
f_BN2VN = 20 # Same as SBC-DBC since no LBC-BP/LBC-DBC/SBC-BP connections in BBP

f_VN2PN = 100
f_VN2MN = 21
f_VN2BN = 20
f_VN2VN = 23 # Same as DBC-DBC since no BP-BP connection in BBP

# Use
use_PN2PN = 0.19 # BBP = 0.05 // Seeman et al (2018) = 0.19
use_PN2MN = 0.094
use_PN2BN = 0.4
use_PN2VN = 0.5

use_MN2PN = 0.3
use_MN2MN = 0.25
use_MN2BN = 0.25
use_MN2VN = 0.24

use_BN2PN = 0.24
use_BN2MN = 0.25
use_BN2BN = 0.25
use_BN2VN = 0.24 # Same as SBC-DBC since no LBC-BP/LBC-DBC/SBC-BP connections in BBP

use_VN2PN = 0.26
use_VN2MN = 0.25
use_VN2BN = 0.24
use_VN2VN = 0.28 # Same as DBC-DBC since no BP-BP connection in BBP

r_NMDA = 2
d_NMDA = 65

#prepost2 is just for test syn, to remove condition where pre==post
syn_params = {
	
	'none' : {'tau_r_AMPA': taur_PN2PN,
				   'tau_d_AMPA': taud_PN2PN,
				   'tau_r_NMDA': r_NMDA,
				   'tau_d_NMDA': d_NMDA,
				   'e': 0,
				   'Dep': d_PN2PN,
				   'Fac': f_PN2PN,
				   'Use': use_PN2PN,
				   'u0':0,
				   'gmax': 0},
	
	'HL5PN1HL5PN1' : {'tau_r_AMPA': taur_PN2PN,
						'tau_d_AMPA': taud_PN2PN,
						'tau_r_NMDA': r_NMDA,
						'tau_d_NMDA': d_NMDA,
						'e': 0,
						'Dep': d_PN2PN,
						'Fac': f_PN2PN,
						'Use': use_PN2PN,
						'u0':0,
						'gmax': con_PN2PN},
	
	'HL5PN1HL5MN1' : {'tau_r_AMPA': taur_PN2MN,
						'tau_d_AMPA': taud_PN2MN,
						'tau_r_NMDA': r_NMDA,
						'tau_d_NMDA': d_NMDA,
						'e': 0,
						'Dep': d_PN2MN,
						'Fac': f_PN2MN,
						'Use': use_PN2MN,
						'u0':0,
						'gmax': con_PN2MN},
	
	'HL5PN1HL5BN1' : {'tau_r_AMPA': taur_PN2BN,
						'tau_d_AMPA': taud_PN2BN,
						'tau_r_NMDA': r_NMDA,
						'tau_d_NMDA': d_NMDA,
						'e': 0,
						'Dep': d_PN2BN,
						'Fac': f_PN2BN,
						'Use': use_PN2BN,
						'u0':0,
						'gmax': con_PN2BN},
	
	'HL5PN1HL5VN1' : {'tau_r_AMPA': taur_PN2VN,
						'tau_d_AMPA': taud_PN2VN,
						'tau_r_NMDA': r_NMDA,
						'tau_d_NMDA': d_NMDA,
						'e': 0,
						'Dep': d_PN2VN,
						'Fac': f_PN2VN,
						'Use': use_PN2VN,
						'u0':0,
						'gmax': con_PN2VN},

	'HL5MN1HL5PN1' : {'tau_r': taur_MN2PN,
						'tau_d': taud_MN2PN,
						'Use': use_MN2PN,
						'Dep': d_MN2PN,
						'Fac': f_MN2PN,
						'e': -80,
						'gmax': con_MN2PN,
						'u0': 0},
	'HL5MN1HL5MN1' : {'tau_r': taur_MN2MN,
						'tau_d': taud_MN2MN,
						'Use': use_MN2MN,
						'Dep': d_MN2MN,
						'Fac': f_MN2MN,
						'e': -80,
						'gmax': con_MN2MN,
						'u0': 0},
	
	'HL5MN1HL5BN1' : {'tau_r': taur_MN2BN,
						'tau_d': taud_MN2BN,
						'Use': use_MN2BN,
						'Dep': d_MN2BN,
						'Fac': f_MN2BN,
						'e': -80,
						'gmax': con_MN2BN,
						'u0': 0},
	'HL5MN1HL5VN1' : {'tau_r': taur_MN2VN,
						'tau_d': taud_MN2VN,
						'Use': use_MN2VN,
						'Dep': d_MN2VN,
						'Fac': f_MN2VN,
						'e': -80,
						'gmax': con_MN2VN,
						'u0': 0},

	'HL5BN1HL5PN1' : {'tau_r': taur_BN2PN,
						'tau_d': taud_BN2PN,
						'Use': use_BN2PN,
						'Dep': d_BN2PN,
						'Fac': f_BN2PN,
						'e': -80,
						'gmax': con_BN2PN,
						'u0': 0},
	'HL5BN1HL5MN1' : {'tau_r': taur_BN2MN,
						'tau_d': taud_BN2MN,
						'Use': use_BN2MN,
						'Dep': d_BN2MN,
						'Fac': f_BN2MN,
						'e': -80,
						'gmax': con_BN2MN,
						'u0': 0},
	
	'HL5BN1HL5BN1' : {'tau_r': taur_BN2BN,
						'tau_d': taud_BN2BN,
						'Use': use_BN2BN,
						'Dep': d_BN2BN,
						'Fac': f_BN2BN,
						'e': -80,
						'gmax': con_BN2BN,
						'u0': 0},
	
	'HL5BN1HL5VN1' : {'tau_r': taur_BN2VN,
						'tau_d': taud_BN2VN,
						'Use': use_BN2VN,
						'Dep': d_BN2VN,
						'Fac': f_BN2VN,
						'e': -80,
						'gmax': con_BN2VN,
						'u0': 0},
	
	'HL5VN1HL5PN1' : {'tau_r': taur_VN2PN,
						'tau_d': taud_VN2PN,
						'Use': use_VN2PN,
						'Dep': d_VN2PN,
						'Fac': f_VN2PN,
						'e': -80,
						'gmax': con_VN2PN,
						'u0': 0},
	'HL5VN1HL5MN1' : {'tau_r': taur_VN2MN,
						'tau_d': taud_VN2MN,
						'Use': use_VN2MN,
						'Dep': d_VN2MN,
						'Fac': f_VN2MN,
						'e': -80,
						'gmax': con_VN2MN,
						'u0': 0},
	'HL5VN1HL5BN1' : {'tau_r': taur_VN2BN,
						'tau_d': taud_VN2BN,
						'Use': use_VN2BN,
						'Dep': d_VN2BN,
						'Fac': f_VN2BN,
						'e': -80,
						'gmax': con_VN2BN,
						'u0': 0},
	'HL5VN1HL5VN1' : {'tau_r': taur_VN2VN,
						'tau_d': taud_VN2VN,
						'Use': use_VN2VN,
						'Dep': d_VN2VN,
						'Fac': f_VN2VN,
						'e': -80,
						'gmax': con_VN2VN,
						'u0': 0}}

mult_syns = {
	'none' : {'loc':0,'scale':0},
	'HL5PN1HL5PN1' : {'loc':Ncont_PN2PN,'scale':0},
	'HL5PN1HL5MN1' : {'loc':Ncont_PN2MN,'scale':0},
	'HL5PN1HL5BN1' : {'loc':Ncont_PN2BN,'scale':0},
	'HL5PN1HL5VN1' : {'loc':Ncont_PN2VN,'scale':0},

	'HL5MN1HL5PN1' : {'loc':Ncont_MN2PN,'scale':0},
	'HL5MN1HL5MN1' : {'loc':Ncont_MN2MN,'scale':0},
	'HL5MN1HL5BN1' : {'loc':Ncont_MN2BN,'scale':0},
	'HL5MN1HL5VN1' : {'loc':Ncont_MN2VN,'scale':0},

	'HL5BN1HL5PN1' : {'loc':Ncont_BN2PN,'scale':0},
	'HL5BN1HL5MN1' : {'loc':Ncont_BN2MN,'scale':0},
	'HL5BN1HL5BN1' : {'loc':Ncont_BN2BN,'scale':0},
	'HL5BN1HL5VN1' : {'loc':Ncont_BN2VN,'scale':0},

	'HL5VN1HL5PN1' : {'loc':Ncont_VN2PN,'scale':0},
	'HL5VN1HL5MN1' : {'loc':Ncont_VN2MN,'scale':0},
	'HL5VN1HL5BN1' : {'loc':Ncont_VN2BN,'scale':0},
	'HL5VN1HL5VN1' : {'loc':Ncont_VN2VN,'scale':0}}

#Stimulation Params
Orientation = 85 # [0-180] Maybe test at 80 vs 100 since there should be strong symmetrical overlap in the Gaussian curves
Pyr_target = 'dend' # 'dend', 'apic', or ['apic', 'dend']
synnum_multiplier = 1 # if 'apic': 1.5, if 'dend': 0.7
# Note that in Bensmaia et al 2008, only ~50% of cells are orientation responsive (depending on area and stimulus type)
PN_params = {'mname':'HL5PN1', 'num_cell': 70, 'num_stim': 10, 'interval': 10, 'stim_type': 'ProbAMPANMDA',
						'start_time': 6500.0, 'delay': 3.0, 'delay_std': 6., 'loc_num': int(45*synnum_multiplier),'loc':Pyr_target,'idx_offset' : 0,
						'syn_params':{'tau_r_AMPA': taur_PN2PN, 'tau_d_AMPA': taud_PN2PN,
						'tau_r_NMDA': r_NMDA, 'tau_d_NMDA': 65, 'e': 0, 'Dep': d_PN2PN,
						'Fac': f_PN2PN, 'Use': use_PN2PN, 'u0':0, 'gmax': con_PN2PN },'Orientation':Orientation}
MN_params = {'mname':'HL5MN1', 'num_cell': 0, 'num_stim': 10, 'interval': 10, 'stim_type': 'ProbAMPANMDA',
						'start_time': 6500.0, 'delay': 3.0, 'delay_std': 6., 'loc_num': 0,'loc':'dend','idx_offset' : 0,
						'syn_params':{'tau_r_AMPA': taur_PN2MN,'tau_d_AMPA': taud_PN2MN,
						'tau_r_NMDA': r_NMDA, 'tau_d_NMDA': 65, 'e': 0, 'Dep': d_PN2MN,
						'Fac': f_PN2MN, 'Use': use_PN2MN, 'u0':0, 'gmax': con_PN2MN },'Orientation':Orientation}
BN_params = {'mname':'HL5BN1', 'num_cell': 20, 'num_stim': 10, 'interval': 10, 'stim_type': 'ProbAMPANMDA',
						'start_time': 6500.0, 'delay': 3.0, 'delay_std': 6., 'loc_num': 30,'loc':'dend','idx_offset' : 0,
						'syn_params':{'tau_r_AMPA': taur_PN2BN, 'tau_d_AMPA': taud_PN2BN,
						'tau_r_NMDA': r_NMDA, 'tau_d_NMDA': 65, 'e': 0, 'Dep': d_PN2BN,
						'Fac': f_PN2BN, 'Use': use_PN2BN, 'u0':0, 'gmax': con_PN2BN },'Orientation':Orientation}
VN_params = {'mname':'HL5VN1', 'num_cell': 10, 'num_stim': 10, 'interval': 10, 'stim_type': 'ProbAMPANMDA',
						'start_time': 6500.0, 'delay': 3.0, 'delay_std': 6., 'loc_num': 30,'loc':'dend','idx_offset' : 0,
						'syn_params':{'tau_r_AMPA': taur_PN2VN, 'tau_d_AMPA': taud_PN2VN,
						'tau_r_NMDA': r_NMDA, 'tau_d_NMDA': 65, 'e': 0, 'Dep': d_PN2VN,
						'Fac': f_PN2VN, 'Use': use_PN2VN, 'u0':0, 'gmax': con_PN2VN },'Orientation':Orientation}
cells_to_stim = [PN_params,MN_params,BN_params,VN_params]
