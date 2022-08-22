# Human L5 Cortical Circuit Model --- Guet-McCreight-et-al.-2022
==============================================================================
Author: Alexandre Guet-McCreight

This is the readme for the model associated with the paper:

Guet-McCreight A, Chameh HM, Mahallati S, Wishart M, Tripathy SJ, Valiante TA, Hay E (2022) Age-dependent increased sag amplitude in human pyramidal neurons dampens baseline cortical activity. Cerebral Cortex.


Network Simulations:
Simulation code associated with the default L5 circuit used throughout the manuscript is in the /L5Circuit/default_circuit/ directory. Additional tests using heterogeneous circuits can be found in the /L5Circuit/heterogeneous/ directory.

To run simulations, install all of the necessary python modules (see lfpy_env.yml), compile the mod files within the mod folder, and submit the simulations in parallel (e.g., see job.sh). Running younger and older circuits is controlled for by changing the agegroup variable in circuit.py to either 'y' or 'o', e.g.:

agegroup = 'y' or 'o'

In job.sh, the 2nd to last number in the mpiexec command (see below) controls the random seed used for the circuit variance (i.e., connection matrix, synapse placement, etc.), while the last number controls the random seed number used for stimulus variance (i.e. Ornstein Uhlenbeck noise and stimulus presynaptic spike train timing).

mpiexec -n 400 python circuit.py 1234 1

All code used for analyzing the circuit simulation results is found in the /L5Circuit Analyses/ directory. These codes perform analyses of multiple simulations using several circuit random seeds (e.g. fig 3) or stimulus random seeds (e.g. fig 5).


Single Cell Optimizations and Simulations:
All code associated with single cell optimizations is found in the /Single Cell Modelling/Optimizations directory. For usage of this code, we recommend starting from the README.md file and associated code in either the /L5Pyr_young_AltMorph_Scinet/ or /L5Pyr_old_AltMorph_Scinet/ directories.

Analysis code of single cell optimization results can be found in the /Single Cell Modelling/Population Analysis/ folder. Current-step single-cell simulation code can be found in the /Single Cell Modelling/Current-Step Simulations/ folder. Code for simulating PSP summation of different connection types can be found in the /Single Cell Modelling/PSP Simulations/ folder.
