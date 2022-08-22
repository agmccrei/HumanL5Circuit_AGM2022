# Simple example of how to load the exported hoc model into NEURON + sanity check
import time
import numpy
import matplotlib.pyplot as plt
from neuron import h, gui
import pprint
import efel
import sys
pp = pprint.PrettyPrinter(indent=4)
TotalStartTime = time.time()

pp.pprint("Initializing...")
sys.stdout.flush()

idx = 0 # Index of model used (same as in 'out_1SimulateModel.py')
cellname = 'interneuron'# Name of template - first specified in 'init_1morphology.py'

h.load_file('work/template' + str(idx) + '.hoc')
h('objref NeuronModel')
h('NeuronModel = new '+cellname+'("experimental_data/")')

h.tstop = 2000
h.dt = 0.025
h.v_init = -81
h.cvode.active(1)

exec('stimobj = h.IClamp(h.'+cellname+'[0].soma[0](0.5))')
stimobj.dur = 1000
stimobj.delay = 270
stimobj.amp = 0.19

input("In the NEURON GUI: Graph -> Voltage axis, and press Enter to simulate...")
h.run()
input("Press Enter to exit...")
