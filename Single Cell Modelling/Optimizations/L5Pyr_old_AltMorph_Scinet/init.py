######### Relevent Links #########
# https://nbviewer.jupyter.org/github/BlueBrain/SimulationTutorials/blob/master/FENS2016/ABI_model/single_cell_model.ipynb
import time
import numpy
import matplotlib.pyplot as plt
from neuron import h, gui
import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pprint
import copy
import efel
import os
import sys
from math import log10, floor
import collections
import pickle
pp = pprint.PrettyPrinter(indent=4)
TotalStartTime = time.time()

Plot_After_Opt = True

pp.pprint("Initializing...")
sys.stdout.flush()

if sys.version_info[0] == 2:
	execfile("init_1morphology.py")
	execfile("init_2mechanisms.py")
	execfile("init_3parameters.py")
	execfile("init_4features.py")
	execfile("init_5recordings.py")
	execfile("init_6fitnesscalculator.py")
	execfile("init_7optimization.py")
	if Plot_After_Opt:
		execfile("init_8plot.py")
elif sys.version_info[0] == 3:
	exec(open("init_1morphology.py").read())
	exec(open("init_2mechanisms.py").read())
	exec(open("init_3parameters.py").read())
	exec(open("init_4features.py").read())
	exec(open("init_5recordings.py").read())
	exec(open("init_6fitnesscalculator.py").read())
	exec(open("init_7optimization.py").read())
	if Plot_After_Opt:
		exec(open("init_8plot.py").read())

TotalTime = time.time() - TotalStartTime
print("Total Time = " + str(TotalTime) + " seconds")

quit()
