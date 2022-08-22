import numpy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pprint
from adjustText import adjust_text
import matplotlib._pylab_helpers
import matplotlib.patheffects as path_effects
import math
import scipy.stats as st
import statsmodels.stats.multitest as mt
import pandas as pd
import glob
import load_swc
import neuron
import seaborn as sns
from neuron import h, gui

filedir = 'HL5PN2.swc'
cell = load_swc.main(filedir)
morph = load_swc.load_swc(filedir,cell)

pp = pprint.PrettyPrinter(indent=4)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)

vars = ['g_pas', 'e_pas', 'gbar_NaTgsomatic', 'gbar_Napsomatic', 'gbar_K_Psomatic', 'gbar_K_Tsomatic', 'gbar_Kv3_1somatic', 'gbar_Imsomatic', 'gbar_SKsomatic', 'decay_CaDynamicssomatic', 'gbar_Ca_HVAsomatic', 'gbar_Ca_LVAsomatic', 'gbar_Ihall', 'gbar_NaTgaxonal', 'gbar_Napaxonal', 'gbar_K_Paxonal', 'gbar_K_Taxonal', 'gbar_Kv3_1axonal', 'gbar_Imaxonal', 'gbar_SKaxonal', 'decay_CaDynamicsaxonal', 'gbar_Ca_HVAaxonal', 'gbar_Ca_LVAaxonal']
labels = [r'$G_{pas}$',r'$E_{pas}$',r'$G_{NaT,s}$',r'$G_{NaP,s}$',r'$G_{K_P,s}$',r'$G_{K_T,s}$',r'$G_{Kv3.1,s}$',r'$G_{M,s}$',r'$G_{SK,s}$',r'$Ca_{Dyn._{\tau,s}}$',r'$G_{Ca_{HVA},s}$',r'$G_{Ca_{LVA},s}$',r'$G_{H}$',r'$G_{NaT,a}$',r'$G_{NaP,a}$',r'$G_{K_P,a}$',r'$G_{K_T,a}$',r'$G_{Kv3.1,a}$',r'$G_{M,a}$',r'$G_{SK,a}$',r'$Ca_{Dyn._{\tau,a}}$',r'$G_{Ca_{HVA},a}$',r'$G_{Ca_{LVA},a}$']
lowerlims = [0.00000001,-100,0,0,0,0,0,0,0,20,0,0,0,0,0,0,0,0,0,0,20,0,0]
upperlims = [0.0002,-72,1,1e-2,1.5,1.5,1.5,5e-4,1.5,1000,1e-4,1e-2,0.0001,1,1e-2,1.5,1.5,1.5,5e-4,1.5,1000,1e-4,1e-2]

dirpath1 = 'young/'
dirpath2 = 'old/'

finalpop_pathy = dirpath1+'finalpop.pkl'
finalpop_patho = dirpath2+'finalpop.pkl'

halloffame_pathy = dirpath1+'halloffame.pkl'
halloffame_patho = dirpath2+'halloffame.pkl'

hist_pathy = dirpath1+'hist.pkl'
hist_patho = dirpath2+'hist.pkl'

logs_pathy = dirpath1+'logs.pkl'
logs_patho = dirpath2+'logs.pkl'

objectives_pathy = dirpath1+'objectives.pkl'
objectives_patho = dirpath2+'objectives.pkl'

finalpopy = pickle.load(open(finalpop_pathy,"rb"))
finalpopo = pickle.load(open(finalpop_patho,"rb"))

halloffamey = pickle.load(open(halloffame_pathy,"rb"))
halloffameo = pickle.load(open(halloffame_patho,"rb"))

histy = pickle.load(open(hist_pathy,"rb"))
histo = pickle.load(open(hist_patho,"rb"))

logsy = pickle.load(open(logs_pathy,"rb"))
logso = pickle.load(open(logs_patho,"rb"))

objectivesy = pickle.load(open(objectives_pathy,"rb"))
objectiveso = pickle.load(open(objectives_patho,"rb"))

# Note: with new scinet code, the objectives are all out of order and not necessarily the same between runs, so need to reconstruct
objectivesy = [obname.name for obname in objectivesy]
objectiveso = [obname.name for obname in objectiveso]

maxBaseSD = 2 # 2 works for young but not all the old models features
maxRMPSD = 2 # same as base SD
maxSagSD = 1 # i.e. higher constraint for sag since that is the focus
voltage_base_Compensator = 2*maxRMPSD # Compensates for having divided the target SD by 2 during Optimization to get better fitness
sag_amplitude_Compensator = 10*maxSagSD # Compensates for having divided the target SD by 10 during Optimization to get better fitness
AP_height_Compensator = [2,6] # since fits are not as good in old
AHP_depth_abs_Compensator = 3 # since fits are not as good in old
AHP_depth_abs_slow_Compensator = 3 # since fits are not as good in old
AHP_slow_time_Compensator = 3 # since fits are not as good in old
AP_width_Compensator = 3 # since fits are not as good in old
Spikecount_Compensator = 2 # since fits are not as good in old
mean_frequency_Compensator = [0.5,0.8] # since fits are not as good in old

def weightvec(objectives,ag):
	weights = numpy.ones(len(objectives))
	for w in range(0,len(weights)):
		if objectives[w] == 'voltage_base':
			weights[w] = weights[w]*voltage_base_Compensator
		elif objectives[w] == 'sag_amplitude':
			weights[w] = weights[w]*sag_amplitude_Compensator
		elif objectives[w] == 'AP_height':
			weights[w] = weights[w]*AP_height_Compensator[0] if ag == 'y' else weights[w]*AP_height_Compensator[1]
		elif objectives[w] == 'AHP_depth_abs':
			weights[w] = weights[w]*AHP_depth_abs_Compensator
		elif objectives[w] == 'AHP_depth_abs_slow':
			weights[w] = weights[w]*AHP_depth_abs_slow_Compensator
		elif objectives[w] == 'AHP_slow_time':
			weights[w] = weights[w]*AHP_slow_time_Compensator
		elif objectives[w] == 'AP_width':
			weights[w] = weights[w]*AP_width_Compensator
		elif objectives[w] == 'Spikecount':
			weights[w] = weights[w]*Spikecount_Compensator
		elif objectives[w] == 'mean_frequency':
			weights[w] = weights[w]*mean_frequency_Compensator[0] if ag == 'y' else weights[w]*mean_frequency_Compensator[1]
		else:
			weights[w] = weights[w]*maxBaseSD
	
	return weights

weightsy = weightvec(objectivesy,'y')
weightso = weightvec(objectiveso,'o')

print(set(zip(objectivesy,weightsy)))
print(set(zip(objectiveso,weightso)))

bestIndexY = 0
bestIndexO = 0#2

# print params of chosen model
chosen = 0
for i2 in range(0,len(halloffameo[0])):
	print(vars[i2]+': '+ str(halloffameo[chosen][i2]))


totalpopy = []
totalpopo = []
qualityval_y = []
qualityval_o = []

qualitysag_y = []
qualitysag_o = []
qualityrmp_y = []
qualityrmp_o = []
sagidx_y = objectivesy.index("sag_amplitude")
rmpidx_y = objectivesy.index("voltage_base")
sagidx_o = objectiveso.index("sag_amplitude")
rmpidx_o = objectiveso.index("voltage_base")
reject_counts_y = [0 for _ in range(0,len(weightsy))]
reject_counts_o = [0 for _ in range(0,len(weightso))]
# iterate over all models across all generations
print(histy.__dict__['genealogy_history'][1].fitness)
for l in range(1,len(histy.__dict__['genealogy_history'])+1):
	f = histy.__dict__['genealogy_history'][l]
	count = 0
	for t in range(0,len(weightsy)):
		if f.fitness.values[t] < weightsy[t]:
			count += 1
		else:
			reject_counts_y[t] += 1
	
	if count == len(weightsy):
		qualityval_y.append(numpy.sum(f.fitness.values))
		qualitysag_y.append(f.fitness.values[sagidx_y]/10)
		qualityrmp_y.append(f.fitness.values[rmpidx_y]/2)
		totalpopy.append(f)
	else:
		continue

for l in range(1,len(histo.__dict__['genealogy_history'])+1):
	f = histo.__dict__['genealogy_history'][l]
	count = 0
	for t in range(0,len(weightso)):
		if f.fitness.values[t] < weightso[t]:
			count += 1
		else:
			reject_counts_o[t] += 1
	
	if count == len(weightso):
		qualityval_o.append(numpy.sum(f.fitness.values))
		qualitysag_o.append(f.fitness.values[sagidx_o]/10)
		qualityrmp_o.append(f.fitness.values[rmpidx_o]/2)
		totalpopo.append(f)
	else:
		continue

# Remove duplicate models
import itertools
totalpopy.sort()
totalpopy = list(totalpopy for totalpopy,_ in itertools.groupby(totalpopy))
totalpopo.sort()
totalpopo = list(totalpopo for totalpopo,_ in itertools.groupby(totalpopo))

print('Young % rejected:')
print(pp.pformat(list(zip(objectivesy,100*numpy.array(reject_counts_y)/len(histy.__dict__['genealogy_history'])))))
print('Old % rejected:')
print(pp.pformat(list(zip(objectiveso,100*numpy.array(reject_counts_o)/len(histo.__dict__['genealogy_history'])))))

print(len(qualityval_y))
print(len(qualityval_o))

summed_score_y = []
summed_score_o = []
for c, bi in enumerate(totalpopy):
	summed_score_y.append(bi.fitness.values)
for c, bi in enumerate(totalpopo):
	summed_score_o.append(bi.fitness.values)

sorted_totalpopy = [i for _,i in sorted(zip(summed_score_y,totalpopy))]
sorted_totalpopo = [i for _,i in sorted(zip(summed_score_o,totalpopo))]

num_models_to_generate = 100

for c, bi in enumerate(sorted_totalpopy):
	if c > num_models_to_generate-1: break
	txtf = open('highly_ranked_models/biophys_HL5PN1y' + str(c+1) + '.hoc', "w")
	txtf.write('proc biophys_HL5PN1(){\n')
	txtf.write('	forsec $o1.all {\n')
	txtf.write('		insert pas\n')
	txtf.write('		insert Ih\n')
	txtf.write('		Ra = 100\n')
	txtf.write('		cm = 0.9\n')
	txtf.write('		e_pas = ' + str(bi[1]) + '\n')
	txtf.write('		g_pas = ' + str(bi[0]) + '\n')
	txtf.write('		gbar_Ih = ' + str(bi[12]) + '\n')
	txtf.write('		shift1_Ih = 144.76545935424588\n')
	txtf.write('		shift2_Ih = 14.382865335237211\n')
	txtf.write('		shift3_Ih = -28.179477866349245\n')
	txtf.write('		shift4_Ih = 99.18311385307702\n')
	txtf.write('		shift5_Ih = 16.42000098505615\n')
	txtf.write('		shift6_Ih = 26.699880497099517\n')
	txtf.write('	}\n')
	txtf.write('	$o1.distribute_channels("apic","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("dend","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("axon","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	\n')
	txtf.write('	forsec $o1.somatic {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[2]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[3]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[4]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[5]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[6]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[8]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[7]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[10]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[11]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[9]) + '\n')
	txtf.write('	}\n')
	txtf.write('	forsec $o1.axonal {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[13]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[14]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[15]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[16]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[17]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[19]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[18]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[21]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[22]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[20]) + '\n')
	txtf.write('	}\n')
	txtf.write('}\n')
	txtf.write('\n')

for c, bi in enumerate(sorted_totalpopo):
	if c > num_models_to_generate-1: break
	txtf = open('highly_ranked_models/biophys_HL5PN1o' + str(c+1) + '.hoc', "w")
	txtf.write('proc biophys_HL5PN1(){\n')
	txtf.write('	forsec $o1.all {\n')
	txtf.write('		insert pas\n')
	txtf.write('		insert Ih\n')
	txtf.write('		Ra = 100\n')
	txtf.write('		cm = 0.9\n')
	txtf.write('		e_pas = ' + str(bi[1]) + '\n')
	txtf.write('		g_pas = ' + str(bi[0]) + '\n')
	txtf.write('		gbar_Ih = ' + str(bi[12]) + '\n')
	txtf.write('		shift1_Ih = 144.76545935424588\n')
	txtf.write('		shift2_Ih = 14.382865335237211\n')
	txtf.write('		shift3_Ih = -28.179477866349245\n')
	txtf.write('		shift4_Ih = 99.18311385307702\n')
	txtf.write('		shift5_Ih = 16.42000098505615\n')
	txtf.write('		shift6_Ih = 26.699880497099517\n')
	txtf.write('	}\n')
	txtf.write('	$o1.distribute_channels("apic","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("dend","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	$o1.distribute_channels("axon","gbar_Ih",5,0.5,24,950,-285,$o1.soma.gbar_Ih)\n')
	txtf.write('	\n')
	txtf.write('	forsec $o1.somatic {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[2]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[3]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[4]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[5]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[6]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[8]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[7]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[10]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[11]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[9]) + '\n')
	txtf.write('	}\n')
	txtf.write('	forsec $o1.axonal {\n')
	txtf.write('		insert NaTg\n')
	txtf.write('		insert Nap\n')
	txtf.write('		insert K_P\n')
	txtf.write('		insert K_T\n')
	txtf.write('		insert Kv3_1\n')
	txtf.write('		insert SK\n')
	txtf.write('		insert Im\n')
	txtf.write('		insert Ca_HVA\n')
	txtf.write('		insert Ca_LVA\n')
	txtf.write('		insert CaDynamics\n')
	txtf.write('		ek = -85\n')
	txtf.write('		ena = 50\n')
	txtf.write('		gbar_NaTg = ' + str(bi[13]) + '\n')
	txtf.write('		vshiftm_NaTg = 0\n')
	txtf.write('		vshifth_NaTg = 10\n')
	txtf.write('		slopem_NaTg = 9\n')
	txtf.write('		slopeh_NaTg = 6\n')
	txtf.write('		gbar_Nap = ' + str(bi[14]) + '\n')
	txtf.write('		gbar_K_P = ' + str(bi[15]) + '\n')
	txtf.write('		gbar_K_T = ' + str(bi[16]) + '\n')
	txtf.write('		gbar_Kv3_1 = ' + str(bi[17]) + '\n')
	txtf.write('		vshift_Kv3_1 = 0\n')
	txtf.write('		gbar_SK = ' + str(bi[19]) + '\n')
	txtf.write('		gbar_Im = ' + str(bi[18]) + '\n')
	txtf.write('		gbar_Ca_HVA = ' + str(bi[21]) + '\n')
	txtf.write('		gbar_Ca_LVA = ' + str(bi[22]) + '\n')
	txtf.write('		gamma_CaDynamics = 0.0005\n')
	txtf.write('		decay_CaDynamics = ' + str(bi[20]) + '\n')
	txtf.write('	}\n')
	txtf.write('}\n')
	txtf.write('\n')

qualityval_y = numpy.array(qualityval_y)
qualityval_o = numpy.array(qualityval_o)
fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(qualityval_y,50,facecolor='k', alpha=0.5,label='Young')
ax.hist(qualityval_o,50,facecolor='darkred', alpha=0.5,label='Old')
bottom, top = ax.get_ylim()
ax.set_ylim((bottom,top))
ax.set_xlabel('Sum of Standard Deviations')
ax.set_ylabel('Model Count')
fig.tight_layout()
fig.savefig('PLOTfiles/Quality.pdf', bbox_inches='tight')
fig.savefig('PLOTfiles/Quality.png', bbox_inches='tight')
plt.close(fig)

print(halloffameo.__dict__['items'][bestIndexO].__dict__['fitness'].__dict__['wvalues'])
bestysag = abs(halloffamey.__dict__['items'][bestIndexY].__dict__['fitness'].__dict__['wvalues'][sagidx_y]/10)
bestyrmp = abs(halloffamey.__dict__['items'][bestIndexY].__dict__['fitness'].__dict__['wvalues'][rmpidx_y]/2)
bestosag = abs(halloffameo.__dict__['items'][bestIndexO].__dict__['fitness'].__dict__['wvalues'][sagidx_o]/10)
bestormp = abs(halloffameo.__dict__['items'][bestIndexO].__dict__['fitness'].__dict__['wvalues'][rmpidx_o]/2)

qualitysag_y = numpy.array(qualitysag_y)
qualitysag_o = numpy.array(qualitysag_o)
qualityrmp_y = numpy.array(qualityrmp_y)
qualityrmp_o = numpy.array(qualityrmp_o)
(y_r, y_p) = st.pearsonr(qualityrmp_y, qualitysag_y)
(o_r, o_p) = st.pearsonr(qualityrmp_o, qualitysag_o)
yfit = numpy.polyfit(qualityrmp_y, qualitysag_y, 1)
ofit = numpy.polyfit(qualityrmp_o, qualitysag_o, 1)
statsy = 'R = ' + str(numpy.around(y_r,2)) + '; p = ' + str(numpy.around(y_p,4))
statso = 'R = ' + str(numpy.around(o_r,2)) + '; p = ' + str(numpy.around(o_p,4))

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(qualityrmp_y,qualitysag_y,color='k', alpha=0.7,label=statsy)
ax.scatter(qualityrmp_o,qualitysag_o,color='darkred', alpha=0.7,label=statso)
ax.scatter(bestyrmp,bestysag,s=500,linewidths=2,edgecolors='k',color='darkgray', alpha=1)
ax.scatter(bestormp,bestosag,s=500,linewidths=2,edgecolors='darkred',color='mistyrose', alpha=1)
for i2 in range(0,len(halloffamey.__dict__['items'])):
	ysag = abs(halloffamey.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][sagidx_y]/10)
	yrmp = abs(halloffamey.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][rmpidx_y]/2)
	osag = abs(halloffameo.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][sagidx_o]/10)
	ormp = abs(halloffameo.__dict__['items'][i2].__dict__['fitness'].__dict__['wvalues'][rmpidx_o]/2)
	text1 = ax.text(yrmp,ysag,str(i2+1),color='white',horizontalalignment='center',verticalalignment='center',fontsize=20)
	text2 = ax.text(ormp,osag,str(i2+1),color='mistyrose',horizontalalignment='center',verticalalignment='center',fontsize=20)
	text1.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
					   path_effects.Normal()])
	text2.set_path_effects([path_effects.Stroke(linewidth=3, foreground='darkred'),
					   path_effects.Normal()])
# plt.plot(qualityrmp_y,qualityrmp_y*yfit[0]+yfit[1],'k')
# plt.plot(qualityrmp_o,qualityrmp_o*ofit[0]+ofit[1],'darkred')
# leg = plt.legend(fontsize=10,loc='upper right', handlelength=0, handletextpad=0, fancybox=True)
bottom, top = ax.get_ylim()
ax.set_ylim((bottom,top))
ax.set_xlabel('Resting Potential Error (SD)')
ax.set_ylabel('Sag Amplitude Error (SD)')
fig.tight_layout()
fig.savefig('PLOTfiles/Quality_SagVsRMP.pdf', dpi=300, bbox_inches='tight',transparent=True)
fig.savefig('PLOTfiles/Quality_SagVsRMP.png', dpi=300, bbox_inches='tight',transparent=True)
plt.close(fig)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)

print(len(totalpopy))
print(len(totalpopo))

finalpopy = numpy.transpose(totalpopy)
finalpopo = numpy.transpose(totalpopo)

halloffamey = numpy.transpose(halloffamey)
halloffameo = numpy.transpose(halloffameo)

print('GHy = ' + str(halloffamey[12][bestIndexY]))
print('GHo = ' + str(halloffameo[12][bestIndexO]))

dists = numpy.linspace(0,1700,1000)
GHy_apics = halloffamey[12][bestIndexY]*(0.5+(24/(1+numpy.exp((dists-950)/(-285)))))
GHo_apics = halloffameo[12][bestIndexO]*(0.5+(24/(1+numpy.exp((dists-950)/(-285)))))
print('Y/O GH Ratio at soma = ' + str(GHo_apics[0]/GHy_apics[0]))
print('Y/O GH Ratio at distal apical = ' + str(GHo_apics[-1]/GHy_apics[-1]))
fig, ax = plt.subplots(1, figsize=(8, 8), facecolor='white')
ax.fill_between(dists,GHo_apics,color='r',alpha=0.7)
ax.fill_between(dists,GHy_apics,color='k',alpha=0.9)
ax.set_ylabel(r'Model G$_H$ (S/cm$^2$)')
ax.set_xlabel(r'Distance from Soma ($\mu$m)')
ax.set_xlim(0,1700)
ax.set_ylim(0,0.00065)
plt.savefig('PLOTfiles/GHvsDist.pdf', bbox_inches='tight', dpi=300)
plt.savefig('PLOTfiles/GHvsDist.png', bbox_inches='tight', dpi=300)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()
# halloffamey = numpy.concatenate((halloffamey,[GHy_apic]))
# halloffameo = numpy.concatenate((halloffameo,[GHo_apic]))
# vars.append('gbar_Ihapic')
# labels.append(r'$G_{H,apic_{'+str(dist2test)+'\mum}}$')

gen_numbersy = logsy.select('gen')
gen_numberso = logso.select('gen')
min_fitnessy = numpy.array(logsy.select('min'))
min_fitnesso = numpy.array(logso.select('min'))
max_fitnessy = logsy.select('max')
max_fitnesso = logso.select('max')
mean_fitnessy = numpy.array(logsy.select('avg'))
mean_fitnesso = numpy.array(logso.select('avg'))
std_fitnessy = numpy.array(logsy.select('std'))
std_fitnesso = numpy.array(logso.select('std'))

fig, ax = plt.subplots(1, figsize=(8, 8), facecolor='white')

stdminus = mean_fitnessy - std_fitnessy
stdplus = mean_fitnessy + std_fitnessy

ax.plot(
	gen_numbersy,
	mean_fitnessy,
	color='black',
	linewidth=2,
	label='Population Average')

ax.fill_between(
	gen_numbersy,
	min_fitnessy,
	max_fitnessy,
	color='black',
	alpha=0.4,
	linewidth=2,
	label=r'Population Standard Deviation')

stdminus = mean_fitnesso - std_fitnesso
stdplus = mean_fitnesso + std_fitnesso

ax.plot(
	gen_numberso,
	mean_fitnesso,
	color='r',
	linewidth=2,
	label='Population Average')

ax.fill_between(
	gen_numberso,
	min_fitnesso,
	max_fitnesso,
	color='r',
	alpha=0.4,
	linewidth=2,
	label=r'Population Standard Deviation')

# ax.plot([min(gen_numbersy) - 1, max(gen_numbersy) + 1],[numpy.max(qualityval_y),numpy.max(qualityval_y)],'k',ls='dashed')
# ax.plot([min(gen_numberso) - 1, max(gen_numberso) + 1],[numpy.max(qualityval_o),numpy.max(qualityval_o)],'r',ls='dashed')
ax.set_xlim(min(gen_numbersy) - 1, max(gen_numbersy) + 1)
ax.set_xlabel('Generation #')
ax.set_ylabel('# Standard Deviations')
ax.set_ylim([10, 100000])
ax.set_yscale('log')
plt.savefig('PLOTfiles/Performance.pdf', bbox_inches='tight', dpi=300)
plt.savefig('PLOTfiles/Performance.png', bbox_inches='tight', dpi=300)
plt.gcf().clear()
plt.cla()
plt.clf()
plt.close()


texts1 = []
texts2 = []
xs = []
ys = []
ratios = []

fig, ax = plt.subplots(figsize=(15,14))
ax.set_xlabel('Old Model',fontsize=26)
ax.set_ylabel('Young Model',fontsize=26)
line1 = ax.plot(numpy.array([1e-7,1e+3]),numpy.array([1e-7,1e+3]),ls='dashed',color='k')
for i in range(0,len(halloffameo)):
	xs.append(halloffameo[i][bestIndexO])
	ys.append(halloffamey[i][bestIndexY])
	ratios.append(halloffamey[i][bestIndexY]/halloffameo[i][bestIndexO])
	if halloffameo[i][bestIndexO]>halloffamey[i][bestIndexY]:
		texts1.append(plt.text(halloffameo[i][bestIndexO], halloffamey[i][bestIndexY], labels[i]))
	else:
		texts2.append(plt.text(halloffameo[i][bestIndexO], halloffamey[i][bestIndexY], labels[i]))

scatter1 = ax.loglog(xs, ys, 'o', markersize=6, color='k')
ax.loglog(halloffameo[12][bestIndexO],halloffamey[12][bestIndexY],'o',fillstyle='none', markersize=14, color='g')
plots=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
ax.grid(True)
ax.set_xlim(1e-7,1e+3)
ax.set_ylim(1e-7,1e+3)

expand = (1.2, 1.2)
forces = 0.4
adjust_text(texts1, precision=0.0000000001, ax=ax, autoalign='xy',
	expand_text=expand, expand_points=expand, expand_objects=expand, expand_align=expand,
	force_text=forces, force_points=forces, force_objects=forces,
	add_objects=line1, lim=99999,
	only_move={'points':'xy', 'texts':'xy', 'objects':'x'},
	arrowprops=dict(arrowstyle="wedge", color='dimgray', lw=1))
expand = (1.2, 1.2)
forces = 0.4
adjust_text(texts2, precision=0.0000000001, ax=ax, autoalign='xy',
	expand_text=expand, expand_points=expand, expand_objects=expand, expand_align=expand,
	force_text=forces, force_points=forces, force_objects=forces,
	add_objects=line1, lim=99999,
	only_move={'points':'xy', 'texts':'xy', 'objects':'y'},
	arrowprops=dict(arrowstyle="wedge", color='dimgray', lw=1))

fig.savefig('PLOTfiles/loglog_params.pdf')
fig.savefig('PLOTfiles/loglog_params.png')
plt.close(fig)

half = int(len(labels)/2)
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(2,1,1)
ax.bar(labels[:half+1],ratios[:half+1],facecolor='k',edgecolor='k',linewidth=3,alpha=0.6)
ax.bar(labels[:half+1],ratios[:half+1],facecolor="None",edgecolor='k',linewidth=3,alpha=1)
ax.plot(numpy.array([-1,half+1]),numpy.array([1,1]),ls='dashed',color='r')
ax.set_ylabel('Y/O')
ax.set_yscale('log')
ax.set_xlim(-0.6,half+1-0.4)
ax2 = fig.add_subplot(2,1,2)
barlist2=ax2.bar(labels[half+1:],ratios[half+1:],facecolor='k',edgecolor='k',linewidth=3,alpha=0.6)
ax2.bar(labels[half+1:],ratios[half+1:],facecolor="None",edgecolor='k',linewidth=3,alpha=1)
barlist2[0].set_facecolor('g')
ax2.plot(numpy.array([-1,half]),numpy.array([1,1]),ls='dashed',color='r')
ax2.set_ylabel('Y/O')
ax2.set_yscale('log')
ax2.set_xlim(-0.6,half-0.4)
fig.savefig('PLOTfiles/ratios_params.pdf', bbox_inches='tight')
fig.savefig('PLOTfiles/ratios_params.png', bbox_inches='tight')
plt.close(fig)


# Create Total Conductance List
vars_Gtot = ['g_pas', 'gbar_NaTg', 'gbar_Nap', 'gbar_K_P', 'gbar_K_T', 'gbar_Kv3_1', 'gbar_Im', 'gbar_SK', 'gbar_Ca_HVA', 'gbar_Ca_LVA', 'gbar_Ihall']
labels_Gtot = [r'$G_{pas}$',r'$G_{NaT}$',r'$G_{NaP}$',r'$G_{K_P}$',r'$G_{K_T}$',r'$G_{Kv3.1}$',r'$G_{M}$',r'$G_{SK}$',r'$G_{Ca_{HVA}}$',r'$G_{Ca_{LVA}}$',r'$G_{H}$']

lowerlims2 = [0.00000001,0,0,0,0,0,0,0,0,0,0]
upperlims2 = [0.0002,1,1e-2,1.5,1.5,1.5,5e-4,1.5,1e-4,1e-2,0.0001]
lowerlims_Gtot = []
upperlims_Gtot = []
lowerlims_Gsoma = []
upperlims_Gsoma = []
# h('access soma')
h('distance()')
for i in range(0,len(lowerlims2)):
	h('Gval1 = 0')
	h('Gval2 = 0')
	h.Gval1 = lowerlims2[i]
	h.Gval2 = upperlims2[i]
	h('totG1 = 0')
	h('totG2 = 0')
	if (vars_Gtot[i] == 'g_pas'):
		h('forall for (x,0) totG1 += (area(x)*Gval1)*(1e+6/1e+8)')
		h('forall for (x,0) totG2 += (area(x)*Gval2)*(1e+6/1e+8)')
	elif (vars_Gtot[i] == 'gbar_Ihall'):
		h('forall for (x,0) totG1 += (area(x)*Gval1)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
		h('forall for (x,0) totG2 += (area(x)*Gval2)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
	else:
		h('forsec "soma" for (x,0) totG1 += (area(x)*Gval1)*(1e+6/1e+8)')
		h('forsec "axon" for (x,0) totG1 += (area(x)*Gval1)*(1e+6/1e+8)')
		h('forsec "soma" for (x,0) totG2 += (area(x)*Gval2)*(1e+6/1e+8)')
		h('forsec "axon" for (x,0) totG2 += (area(x)*Gval2)*(1e+6/1e+8)')
	lowerlims_Gtot.append(h.totG1)
	upperlims_Gtot.append(h.totG2)
	lowerlims_Gsoma.append(h.Gval1)
	upperlims_Gsoma.append(h.Gval2)

print(lowerlims_Gtot)
print(upperlims_Gtot)

Gtotsy = []
Gtotso = []
Gsomay = []
Gsomao = []
count = 0
for i in range(0,len(finalpopy)-10):
	if ((vars[i] == 'e_pas') | (vars[i] == 'decay_CaDynamicssomatic') | (vars[i] == 'decay_CaDynamicsaxonal')):
		continue
	elif vars[i] == 'g_pas':
		Gsomay.append([])
		Gtotsy.append([])
		for Gval in finalpopy[i]:
			Gsomay[count].append(Gval)
			h('Gval = 0')
			h.Gval = Gval
			h('totG = 0')
			h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)')
			Gtotsy[count].append(h.totG)
		Gsomao.append([])
		Gtotso.append([])
		for Gval in finalpopo[i]:
			Gsomao[count].append(Gval)
			h('Gval = 0')
			h.Gval = Gval
			h('totG = 0')
			h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)')
			Gtotso[count].append(h.totG)
		count += 1
	elif vars[i] == 'gbar_Ihall':
		Gsomay.append([])
		Gtotsy.append([])
		for Gval in finalpopy[i]:
			Gsomay[count].append(Gval)
			h('Gval = 0')
			h.Gval = Gval
			h('totG = 0')
			h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
			Gtotsy[count].append(h.totG)
		Gsomao.append([])
		Gtotso.append([])
		for Gval in finalpopo[i]:
			Gsomao[count].append(Gval)
			h('Gval = 0')
			h.Gval = Gval
			h('totG = 0')
			h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
			Gtotso[count].append(h.totG)
		count += 1
	else:
		Gsomay.append([])
		Gtotsy.append([])
		for j in range(0,len(finalpopy[i])):
			Gsomay[count].append(finalpopy[i][j])
			Gval1 = finalpopy[i][j]
			Gval2 = finalpopy[i+11][j]
			h('Gval1 = 0')
			h('Gval2 = 0')
			h.Gval1 = Gval1
			h.Gval2 = Gval2
			h('totG = 0')
			h('forsec "soma" for (x,0) totG += (area(x)*Gval1)*(1e+6/1e+8)')
			h('forsec "axon" for (x,0) totG += (area(x)*Gval2)*(1e+6/1e+8)')
			Gtotsy[count].append(h.totG)
		Gsomao.append([])
		Gtotso.append([])
		for j in range(0,len(finalpopo[i])):
			Gsomao[count].append(finalpopo[i][j])
			Gval1 = finalpopo[i][j]
			Gval2 = finalpopo[i+11][j]
			h('Gval1 = 0')
			h('Gval2 = 0')
			h.Gval1 = Gval1
			h.Gval2 = Gval2
			h('totG = 0')
			h('forsec "soma" for (x,0) totG += (area(x)*Gval1)*(1e+6/1e+8)')
			h('forsec "axon" for (x,0) totG += (area(x)*Gval2)*(1e+6/1e+8)')
			Gtotso[count].append(h.totG)
		count += 1

Besty = []
Besto = []
Bestsomay = []
Bestsomao = []
for i in range(0,len(halloffamey)-10):
	if ((vars[i] == 'e_pas') | (vars[i] == 'decay_CaDynamicssomatic') | (vars[i] == 'decay_CaDynamicsaxonal')):
		continue
	elif vars[i] == 'g_pas':
		h('Gval = 0')
		h.Gval = halloffamey[i][bestIndexY]
		h('totG = 0')
		h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)')
		Bestsomay.append(h.Gval)
		Besty.append(h.totG)
		h('Gval = 0')
		h.Gval = halloffameo[i][bestIndexO]
		h('totG = 0')
		h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)')
		Bestsomao.append(h.Gval)
		Besto.append(h.totG)
	elif vars[i] == 'gbar_Ihall':
		h('Gval = 0')
		h.Gval = halloffamey[i][bestIndexY]
		h('totG = 0')
		h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
		Bestsomay.append(h.Gval)
		Besty.append(h.totG)
		h('Gval = 0')
		h.Gval = halloffameo[i][bestIndexO]
		h('totG = 0')
		h('forall for (x,0) totG += (area(x)*Gval)*(1e+6/1e+8)*(0.5+(24/(1+exp((distance(x)-950)/(-285)))))')
		Bestsomao.append(h.Gval)
		Besto.append(h.totG)
	else:
		Gval1 = halloffamey[i][bestIndexY]
		Gval2 = halloffamey[i+11][bestIndexY]
		h('Gval1 = 0')
		h('Gval2 = 0')
		h.Gval1 = Gval1
		h.Gval2 = Gval2
		h('totG = 0')
		h('forsec "soma" for (x,0) totG += (area(x)*Gval1)*(1e+6/1e+8)')
		h('forsec "axon" for (x,0) totG += (area(x)*Gval2)*(1e+6/1e+8)')
		Bestsomay.append(h.Gval1)
		Besty.append(h.totG)
		Gval1 = halloffameo[i][bestIndexO]
		Gval2 = halloffameo[i+11][bestIndexO]
		h('Gval1 = 0')
		h('Gval2 = 0')
		h.Gval1 = Gval1
		h.Gval2 = Gval2
		h('totG = 0')
		h('forsec "soma" for (x,0) totG += (area(x)*Gval1)*(1e+6/1e+8)')
		h('forsec "axon" for (x,0) totG += (area(x)*Gval2)*(1e+6/1e+8)')
		Bestsomao.append(h.Gval1)
		Besto.append(h.totG)

print(Besty)
print(Besto)

stats = []
xtickspoints = numpy.linspace(1,len(labels_Gtot),len(labels_Gtot))
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)
for i in range(0,len(Gtotsy)):
	tstat, pval = st.ranksums(Gsomay[i],Gsomao[i])
	effectsize = (numpy.mean(Gsomay[i]) - numpy.mean(Gsomao[i]))/(numpy.std(numpy.concatenate((Gsomay[i],Gsomao[i]))))
	stats.append([vars_Gtot[i],tstat,pval,effectsize])
	
	fpy = numpy.array(Gsomay[i])
	fpo = numpy.array(Gsomao[i])
	normalizedy = (fpy-lowerlims_Gsoma[i])/(upperlims_Gsoma[i]-lowerlims_Gsoma[i])
	normalizedo = (fpo-lowerlims_Gsoma[i])/(upperlims_Gsoma[i]-lowerlims_Gsoma[i])
	
	min_normalizedy = numpy.percentile(normalizedy,5)
	min_normalizedo = numpy.percentile(normalizedo,5)
	max_normalizedy = numpy.percentile(normalizedy,95)
	max_normalizedo = numpy.percentile(normalizedo,95)
	
	median_normalizedy = numpy.median(normalizedy)
	median_normalizedo = numpy.median(normalizedo)
	
	best_normalizedy = (Bestsomay[i]-lowerlims_Gsoma[i])/(upperlims_Gsoma[i]-lowerlims_Gsoma[i]) # young
	best_normalizedo = (Bestsomao[i]-lowerlims_Gsoma[i])/(upperlims_Gsoma[i]-lowerlims_Gsoma[i]) # old
	
	asymmetric_errory = [[median_normalizedy-min_normalizedy], [max_normalizedy-median_normalizedy]]
	asymmetric_erroro = [[median_normalizedo-min_normalizedo], [max_normalizedo-median_normalizedo]]
	
	ax1.errorbar(xtickspoints[i]-0.1, median_normalizedy, yerr=asymmetric_errory, fmt='o', color='k')
	ax1.errorbar(xtickspoints[i]+0.1, median_normalizedo, yerr=asymmetric_erroro, fmt='o', color='r')

ax1.set_ylim(-0.01,1.01)
ax1.set_xticks(xtickspoints)
ax1.set_xticklabels(labels_Gtot, rotation=90)
fig.tight_layout()
fig.savefig('PLOTfiles/normalized_params_Gsoma.pdf')
fig.savefig('PLOTfiles/normalized_params_Gsoma.png')
plt.close(fig)

allpvals = [float(i) for i in numpy.transpose(stats)[2].tolist()]
pval_adj_BO = mt.multipletests(allpvals,method='bonferroni')
pval_adj_BH = mt.multipletests(allpvals,method='fdr_bh')
pval_adj_BO = pval_adj_BO[1]
pval_adj_BH = pval_adj_BH[1]

# Just ad list to dataframe now
df3 = pd.DataFrame(stats, columns=["Parameter", "t-stat", "p-value","Cohen's d"])
df3['p-values adjusted (Bonferroni)'] = pval_adj_BO
df3['p-values adjusted (BH)'] = pval_adj_BH
df3.to_csv('stats_somaG.csv')


stats = []
xtickspoints = numpy.linspace(1,len(labels_Gtot),len(labels_Gtot))
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)
for i in range(0,len(Gtotsy)):
	tstat, pval = st.ranksums(Gtotsy[i],Gtotso[i])
	effectsize = (numpy.mean(Gtotsy[i]) - numpy.mean(Gtotso[i]))/(numpy.std(numpy.concatenate((Gtotsy[i],Gtotso[i]))))
	stats.append([vars_Gtot[i],tstat,pval,effectsize])
	
	fpy = numpy.array(Gtotsy[i])
	fpo = numpy.array(Gtotso[i])
	normalizedy = (fpy-lowerlims_Gtot[i])/(upperlims_Gtot[i]-lowerlims_Gtot[i])
	normalizedo = (fpo-lowerlims_Gtot[i])/(upperlims_Gtot[i]-lowerlims_Gtot[i])
	
	min_normalizedy = numpy.percentile(normalizedy,5)
	min_normalizedo = numpy.percentile(normalizedo,5)
	max_normalizedy = numpy.percentile(normalizedy,95)
	max_normalizedo = numpy.percentile(normalizedo,95)
	
	median_normalizedy = numpy.median(normalizedy)
	median_normalizedo = numpy.median(normalizedo)
	
	best_normalizedy = (Besty[i]-lowerlims_Gtot[i])/(upperlims_Gtot[i]-lowerlims_Gtot[i]) # young
	best_normalizedo = (Besto[i]-lowerlims_Gtot[i])/(upperlims_Gtot[i]-lowerlims_Gtot[i]) # old
	
	asymmetric_errory = [[median_normalizedy-min_normalizedy], [max_normalizedy-median_normalizedy]]
	asymmetric_erroro = [[median_normalizedo-min_normalizedo], [max_normalizedo-median_normalizedo]]
	
	ax1.errorbar(xtickspoints[i]-0.1, median_normalizedy, yerr=asymmetric_errory, fmt='o', color='k')
	ax1.errorbar(xtickspoints[i]+0.1, median_normalizedo, yerr=asymmetric_erroro, fmt='o', color='r')

ax1.set_ylim(-0.01,1.01)
ax1.set_xticks(xtickspoints)
ax1.set_xticklabels(labels_Gtot, rotation=90)
fig.tight_layout()
fig.savefig('PLOTfiles/normalized_params_Gtot.pdf')
fig.savefig('PLOTfiles/normalized_params_Gtot.png')
plt.close(fig)

allpvals = [float(i) for i in numpy.transpose(stats)[2].tolist()]
pval_adj_BO = mt.multipletests(allpvals,method='bonferroni')
pval_adj_BH = mt.multipletests(allpvals,method='fdr_bh')
pval_adj_BO = pval_adj_BO[1]
pval_adj_BH = pval_adj_BH[1]

# Just ad list to dataframe now
df2 = pd.DataFrame(stats, columns=["Parameter", "t-stat", "p-value","Cohen's d"])
df2['p-values adjusted (Bonferroni)'] = pval_adj_BO
df2['p-values adjusted (BH)'] = pval_adj_BH
df2.to_csv('stats_totalG.csv')

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 16}

matplotlib.rc('font', **font)

# Stats and normal tests
stats = []
for i in range(0,len(finalpopy)):
	
	tstat, pval = st.ranksums(finalpopy[i],finalpopo[i])
	effectsize = (numpy.mean(finalpopy[i]) - numpy.mean(finalpopo[i]))/(numpy.std(numpy.concatenate((finalpopy[i],finalpopo[i]))))
	stats.append([vars[i],tstat,pval,effectsize])
	
	plt.hist(finalpopy[i],50,facecolor='k', alpha=0.5,label='Young')
	plt.hist(finalpopo[i],50,facecolor='darkred', alpha=0.5,label='Old')
	bottom, top = plt.ylim()
	plt.plot([halloffamey[i][bestIndexY],halloffamey[i][bestIndexY]],[bottom,top],linestyle='dashed',color='k')
	plt.plot([halloffameo[i][bestIndexO],halloffameo[i][bestIndexO]],[bottom,top],linestyle='dashed',color='darkred')
	plt.ylim((bottom,top))
	# plt.xlim((lowerlims[i],upperlims[i]))
	if ((vars[i] == 'decay_CaDynamicssomatic') or (vars[i] == 'decay_CaDynamicsaxonal')):
		plt.xlabel(labels[i])
	elif (vars[i] == 'e_pas'):
		plt.xlabel(labels[i] + r' (mV)')
	else:
		plt.xlabel(labels[i] + r' (S/cm$^{2}$)')
		plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	
	# plt.ylabel('Count in Final Population')
	if (vars[i] == 'g_pas'):
		plt.legend()
	
	plt.tight_layout()
	plt.savefig('PLOTfiles/' + vars[i] + '.pdf', bbox_inches='tight')
	plt.savefig('PLOTfiles/' + vars[i] + '.png', bbox_inches='tight')
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()

allpvals = [float(i) for i in numpy.transpose(stats)[2].tolist()]
pval_adj_BO = mt.multipletests(allpvals,method='bonferroni')
pval_adj_BH = mt.multipletests(allpvals,method='fdr_bh')
pval_adj_BO = pval_adj_BO[1]
pval_adj_BH = pval_adj_BH[1]

# Just ad list to dataframe now
df = pd.DataFrame(stats, columns=["Parameter", "t-stat", "p-value","Cohen's d"])
df['p-values adjusted (Bonferroni)'] = pval_adj_BO
df['p-values adjusted (BH)'] = pval_adj_BH
df.to_csv('stats.csv')

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)

xtickspoints = numpy.linspace(1,len(labels),len(labels))
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(111)
for i in range(0,len(finalpopy)):
	fpy = finalpopy[i]
	fpo = finalpopo[i]
	normalizedy = (fpy-lowerlims[i])/(upperlims[i]-lowerlims[i])
	normalizedo = (fpo-lowerlims[i])/(upperlims[i]-lowerlims[i])
	
	min_normalizedy = numpy.percentile(normalizedy,5)
	min_normalizedo = numpy.percentile(normalizedo,5)
	max_normalizedy = numpy.percentile(normalizedy,95)
	max_normalizedo = numpy.percentile(normalizedo,95)
	
	median_normalizedy = numpy.median(normalizedy)
	median_normalizedo = numpy.median(normalizedo)
	
	best_normalizedy = (halloffamey[i][bestIndexY]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # young
	best_normalizedo = (halloffameo[i][bestIndexO]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # old
	
	asymmetric_errory = [[median_normalizedy-min_normalizedy], [max_normalizedy-median_normalizedy]]
	asymmetric_erroro = [[median_normalizedo-min_normalizedo], [max_normalizedo-median_normalizedo]]
	
	ax1.errorbar(xtickspoints[i]-0.1, median_normalizedy, yerr=asymmetric_errory, fmt='o', color='k')
	ax1.errorbar(xtickspoints[i]+0.1, median_normalizedo, yerr=asymmetric_erroro, fmt='o', color='r')

ax1.set_xticks(xtickspoints)
ax1.set_xticklabels(labels, rotation=90)
fig.tight_layout()
fig.savefig('PLOTfiles/normalized_params.pdf')
fig.savefig('PLOTfiles/normalized_params.png')
plt.close(fig)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)

xtickspoints = numpy.array([1])
fig = plt.figure(figsize=(4,4))
ax1 = fig.add_subplot(111)
i = 12

fpy = finalpopy[i]
fpo = finalpopo[i]
normalizedy = (fpy-lowerlims[i])/(upperlims[i]-lowerlims[i])
normalizedo = (fpo-lowerlims[i])/(upperlims[i]-lowerlims[i])

min_normalizedy = numpy.percentile(normalizedy,5)
min_normalizedo = numpy.percentile(normalizedo,5)
max_normalizedy = numpy.percentile(normalizedy,95)
max_normalizedo = numpy.percentile(normalizedo,95)

best_normalizedy = (halloffamey[i][bestIndexY]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # young
best_normalizedo = (halloffameo[i][bestIndexO]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # old

asymmetric_errory = [[best_normalizedy-min_normalizedy], [max_normalizedy-best_normalizedy]]
asymmetric_erroro = [[best_normalizedo-min_normalizedo], [max_normalizedo-best_normalizedo]]

ax1.errorbar(xtickspoints-0.1, best_normalizedy, yerr=asymmetric_errory, fmt='o', color='k')
ax1.errorbar(xtickspoints+0.1, best_normalizedo, yerr=asymmetric_erroro, fmt='o', color='r')

ax1.set_xticks([xtickspoints-0.1,xtickspoints-0.1])
ax1.set_xlim(0.8,1.2)
ax1.set_xticklabels('', rotation=90)
fig.tight_layout()
fig.savefig('PLOTfiles/normalizedIH_params.pdf', bbox_inches='tight', dpi=300, transparent=True)
fig.savefig('PLOTfiles/normalizedIH_params.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)

xtickspoints = numpy.array([1])
fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
i = 12

fpy = finalpopy[i]
fpo = finalpopo[i]
normalizedy = (fpy-lowerlims[i])/(upperlims[i]-lowerlims[i])
normalizedo = (fpo-lowerlims[i])/(upperlims[i]-lowerlims[i])

min_normalizedy = numpy.percentile(normalizedy,5)
min_normalizedo = numpy.percentile(normalizedo,5)
max_normalizedy = numpy.percentile(normalizedy,95)
max_normalizedo = numpy.percentile(normalizedo,95)

best_normalizedy = (halloffamey[i][bestIndexY]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # young
best_normalizedo = (halloffameo[i][bestIndexO]-lowerlims[i])/(upperlims[i]-lowerlims[i]) # old

asymmetric_errory = [[best_normalizedy-min_normalizedy], [max_normalizedy-best_normalizedy]]
asymmetric_erroro = [[best_normalizedo-min_normalizedo], [max_normalizedo-best_normalizedo]]

yvals = normalizedy[(normalizedy>min_normalizedy) & (normalizedy<max_normalizedy)]
ovals = normalizedo[(normalizedo>min_normalizedo) & (normalizedo<max_normalizedo)]

parts1 = ax1.violinplot(yvals, positions=xtickspoints-0.25, showmeans=False, showmedians=False,
		showextrema=False)
parts2 = ax1.violinplot(ovals, positions=xtickspoints+0.25, showmeans=False, showmedians=False,
		showextrema=False)

for pc in parts1['bodies']:
	pc.set_facecolor('dimgray')
	pc.set_edgecolor('black')
	pc.set_alpha(1)
for pc in parts2['bodies']:
	pc.set_facecolor('r')
	pc.set_edgecolor('r')
	pc.set_alpha(0.7)

xsesy = numpy.random.rand(len(yvals))*0.4+0.55
xseso = numpy.random.rand(len(ovals))*0.4+1.05
ax1.scatter(xsesy,yvals,s=1,c='k',marker='o')
ax1.scatter(xseso,ovals,s=1,c='k',marker='o')
ax1.scatter(xtickspoints-0.25,best_normalizedy,s=60,c='gold',marker='o',linewidths=0.5,edgecolors='k')
ax1.scatter(xtickspoints+0.25,best_normalizedo,s=60,c='gold',marker='o',linewidths=0.5,edgecolors='k')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xticks([xtickspoints-0.25,xtickspoints+0.25])
ax1.set_xticklabels('', rotation=90)
ax1.set_xlim(0.4,1.6)
fig.tight_layout()
fig.savefig('PLOTfiles/normalizedIHviolin_params.pdf', bbox_inches='tight', dpi=300, transparent=True)
fig.savefig('PLOTfiles/normalizedIHviolin_params.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig)
