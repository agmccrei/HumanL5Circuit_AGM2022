# Neuron_Optimizations
==============================================================================
Author: Alexandre Guet-McCreight

Neuron Optimization Using Single Neuron or Population Data.
This code performs neuron optimizations (using bluepyopt: http://bluepyopt.readthedocs.io/en/latest) with either cell type data from Allen Brain Institute or manually entered population data.

This code has been tested using an example L5 pyramidal neuron with the following conditions:
cellname = 'pyramidal'
morphname = 'experimental_data/H16.06.013.11.18.02_696537235_m.swc'
target_feature_type = 'Manual'
c_soma = [-0.40, -0.05, 0.10, 0.20, 0.30] # nA
c_axon = [0.30] # nA

... as well as an example putative L1 neurogliaform interneuron with the following conditions:
cellname = 'interneuron'
morphname = 'experimental_data/H17.03.009.11.11.01_650039094_m.swc'
target_feature_type = 'Automatic'
c_soma = [-0.07, -0.03, 0.04, 0.12, 0.19] # nA
c_axon = [0.19] # nA

Both with 400 generations and 400 offspring per generation. See the 'examples' folder for plots and output model files from these tests (not fine-tuned example optimizations, so results could be better, but just to show some example outputs). The following outlines how to set all of these variables throughout the different files.

-----------------------------------
Step 0: Acquiring & Converting Data
-----------------------------------
1. If optimizing to single neuron voltage traces:
Obtain data from a cell type subpage on Allen Brain Institute.
Example: https://celltypes.brain-map.org/experiment/electrophysiology/596887423

On this subpage, download the electrophysiology data file (extension: .nwb). Also click to view the morphology and then download the reconstruction file (extension: .swc). Move both of these files into the 'experimental_data' folder.

Also note on the webpage the sweep numbers that you want to optimize your model to (colored squares).

Before the optimization, open the 'init_0singlecell_ABIdataconvert.py' file and enter the sweep numbers for the hyperpolarizing ('passteps' variable) and depolarizing ('actsteps' variable) step numbers that you want to optimize to. Also edit the 'nwb_data' variable to input the correct nwb filename.

Run the conversion with the following command:
python init_0singlecell_ABIdataconvert.py

This should create new files in your 'experimental_data' folder, which includes generating pkl files containing the selected junction potential corrected traces that get inputted into the optimization:
'active.pkl' and 'passive.pkl'

2. If optimizing to population data (or if you'd rather enter the feature values manually):
Depending on your dataset, it is best (in the context of this code and bluepyopt) to extract e-feature values from the dataset using the efel module since bluepyopt directly uses the efel module to the calculate feature values used for evaluating model fitness.

The 'init_0populationfeatures_example.py' shows some example code for extracting mean +/- SD feature values from a dataset on Github (https://github.com/stripathy/valiante_ih/tree/master/summary_tables).

--------------------------
Step 1: Loading Morphology
--------------------------
Edit the 'cellname' variable to either 'pyramidal' or 'interneuron'. That is, set it to 'pyramidal' if the morphology contains apical dendrites, as this will allow the extra initialization of apical dendritic parameters (passive + Ih).

Edit 'init_1morphology.py' to input the correct swc filename to the 'morphname' variable.

Also, specify if using a custom replace_axon function using the 'CustomAxonReplacement' variable (if True, uses the replace_axon function defined in this file instead of the default bluepyopt replace_axon method). Note that the python and hoc versions defined here need to be equivalent for correctly exporting models to a hoc file, so any edits to the python function should be equivalently written in the hoc version.

----------------------------
Step 2: Selecting Mechanisms
----------------------------
Review the mechanisms defined in 'init_2mechanisms.py' and which morphological sections the mechanisms are placed in. Edit as necessary.

----------------------------
Step 3: Selecting Parameters
----------------------------
Review the frozen and free parameters as well as their ranges in 'init_3parameters.py', and edit as necessary.

---------------------------------------
Step 4: Selecting Experimental Features
---------------------------------------
Edit the following variables in 'init_4features.py':

a. Specify if calculating features automatically from inputted voltage traces from a single cell ('target_feature_type = 'Automatic''), or if features are to be inputted manually ('target_feature_type = 'Manual'').
b. c_soma: enter the corresponding current amplitudes
c. c_axon: enter copies of any somatic current amplitudes to include feature targets in non-somatic compartments (e.g. axon at the highest spiking step).

1. If optimizing to single neuron voltage traces ('target_feature_type = 'Automatic''), edit the 1st half of the get_features() function (i.e. starting from 'if target_feature_type == 'Automatic''):
Review the features used for each step (must be efel features: https://efel.readthedocs.io/en/latest/) and edit the standard deviation (sd) values, as desired. Because there is no population standard deviation in this context, smaller sd values simply add more weight to chosen features. If not specified, sd = abs(mean*0.1).

2. If optimizing to population data ('target_feature_type = 'Manual''), edit the 2nd half of the get_features() function (i.e. starting from 'elif target_feature_type == 'Manual''):
Manually enter the population means and standard deviation values obtained for each feature (again, must be efel features). Indices of the 'features_values_mean' and 'features_values_sd' variables are in the same order as the indices of the 'active_steps' variable. Different efel features can be added/removed, if desired.

--------------------------------------------------------
Step 5: Selecting Recording Sites and Stimulus Protocols
--------------------------------------------------------
Review the recording sites and stimulus protocols in 'init_5recordings.py' and edit as necessary. Many of the variables are first initialized in earlier steps/files (e.g. 'rec1name', 'rec2name', 'c_soma', 'amplitude', 'stimstart', 'stimend', etc.), so it's important to keep things consistent.

------------------------------------
Step 6: Preparing Fitness Calculator
------------------------------------
Review the methods used for evaluating model fitness in 'init_6fitnesscalculator.py'. The default code groups together features of the same name across steps and takes the maximum error from each, which are then summed together to generate the fitness score (i.e. where smaller = better).

-------------------------------
Step 7: Setting up Optimization
-------------------------------
In 'init_7optimization.py', review the optimization variables, namely the number of offspring per generation ('offss'), the number of generations ('ngens'), the crossover probability ('cxpbs'), the mutation probability ('mutpbs'), the learning rate ('etas'), and the algorithm that you want to use ('SELECTOR'). If debug_run is set to True, then only 10 offspring per generation and 10 generations are used.

---------------------------
Step 8: Setting up Plotting
---------------------------
If using the plotting functions that I have set to run by default (i.e. in 'init.py', set 'Plot_After_Opt = True' -> this toggles the code to also run 'init_8plot.py').

-------------------------------------------------
Step 9: Running an Optimization Job (In Parallel)
-------------------------------------------------
When ready, load the main folder and subdirectories to scinet and run the optimization using one of the following commands:
sbatch job_debug.sh
sbatch job.sh

The job_debug.sh file is used for testing code -> this submits the job to the debug queue instead of the normal queue. If using this option, also set 'debug_run' in 'init_7optimization.py' to True, which calls for smaller numbers of offspring and generations. Otherwise, use job.sh for full runs.

The general workflow of the .sh scripts are to 1) create an ipython profile in the $SCRATCH/.ipython/ directory, 2) start the ipcontroller and ipengines manually, and 3) launch the job. In 'init_7optimization.py', the ipython profile gets inputted to ipyparallel's Client() function, which is used to create the map function for performing the optimization in parallel.

Note: If not already installed on scinet, you should be able to simply pip install the necessary python modules to your existing LFPy virtual environment, namely:
bluepyopt
efel
ipyparallel

Review installation instructions for these modules online.
https://bluepyopt.readthedocs.io/en/latest/
https://efel.readthedocs.io/en/latest/installation.html
https://ipyparallel.readthedocs.io/en/latest/

-----------------------------------------
Step 10: Post-Analysis & Model Generation
-----------------------------------------
Post-optimization analyses can be run using the following files. Note that mod files may need to be compiled using the 'nrnivmodl modfiles/' command before running these scripts.

1. 'python out_1SimulateModel.py': Loads a select model from the top 10 models ('idx' variable -> 0 to 9), exports it to a hoc file (in the 'work/' folder), and simulates and plots the model.

2. 'python out_2SimulateHocModel.py': Run this as a quick sanity check for making sure that the exported hoc file outputs match with the out_1SimulateModel.py outputs. Desired simulation protocol is entered manually and does not rely on any of the bluepyopt scripts. Note that the program pauses for you to open a voltage graph from the NEURON GUI (Graph -> Voltage Axis), and continues to run the simulation once you press the Enter key.

3. 'python out_3AnalyzeResults.py': Runs some example analyses and generates plots of highly ranked models that were generated throughout the optimizations. Some tinkering with the threshold SD values is usually necessary to run this code to completion.

# Neuron_Optimizations
