########################
### Run Optimization ###
########################
# Optimization parameters
debug_run = False # Sets ngens and offss to 10 each

if debug_run:
	offss = 10
	ngens = 10
else:
	offss = 400
	ngens = 300
cxpbs = 0.1
mutpbs = 0.35
etas = 10
SEED = 2400
SELECTOR = 'IBEA' # IBEA or NSGA2

# Access ipython parallel profile
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", required=True,
		help="Name of IPython profile to use")
args = parser.parse_args()

from ipyparallel import Client
rc = Client(profile=args.profile,profile_dir='$SCRATCH/.ipython/profile_'+args.profile,ipython_dir='$SCRATCH/.ipython/')

sys.stderr.write('Using ipyparallel with {0} engines'.format(len(rc)))
sys.stderr.flush()
sys.stderr.write("\n")
sys.stderr.flush()

# If using custom replace_axon, apply the patch to all ipyparallel engines
def rerunmorph_remote():
	import sys
	import bluepyopt.ephys as ephys
	if sys.version_info[0] == 2:
		execfile("init_1morphology.py")
	elif sys.version_info[0] == 3:
		exec(open("init_1morphology.py").read())

dview = rc[:]
if CustomAxonReplacement:
	dview.apply_sync(rerunmorph_remote)

# Define map function
lview = rc.load_balanced_view()

def mapper(func, it):
	start_time = time.time()
	ret = lview.map_sync(func, it)
	sys.stderr.write('Generation took {0}'.format(time.time() - start_time))
	sys.stderr.flush()
	sys.stderr.write("\n")
	sys.stderr.flush()
	return ret

map_function = mapper


# Create and start optimization
optimisation = bpop.optimisations.DEAPOptimisation(
		evaluator=cell_evaluator,
		map_function = map_function,
		cxpb = cxpbs,
		mutpb = mutpbs,
		eta = etas,
		offspring_size = offss,
		selector_name = SELECTOR,
		seed=SEED)

finalpop, halloffame, logs, hist = optimisation.run(max_ngen=ngens)

##### Save Population #####
f = open("results/finalpop.pkl","wb")
pickle.dump(finalpop,f,protocol=2)
f.close()
f = open("results/halloffame.pkl","wb")
pickle.dump(halloffame,f,protocol=2)
f.close()
f = open("results/logs.pkl","wb")
pickle.dump(logs,f,protocol=2)
f.close()
f = open("results/hist.pkl","wb")
pickle.dump(hist,f,protocol=2)
f.close()
