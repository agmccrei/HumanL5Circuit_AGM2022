################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
from __future__ import division
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
import scipy.fftpack
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import itertools
import random
import pandas as pd

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, accuracy_score)

N_PNs = 700 # Number of PNs
N_PNs_stim = int(N_PNs/2) # Number of stimulated PNs
N_seeds = 80
rndinds = np.linspace(1,N_seeds,N_seeds, dtype=int)
num2train = 15 # samples per angle used for training
num2test = N_seeds-num2train # samples per angle used for actual tests
NPermutes = 300 # Number of times to randomly permute the samples for different training/validation/tests
NFeats = 100 # Number of features to include - Only matters if SubFeats = True

# Only one can be True
Metric = 'area100' # options: 'area50', 'area100', 'area200', 'max50', 'max100', 'max200', 'num50', 'num100', 'num200', 'areahf100', 'maxhf100'
AllFeats = True
StimFeats = False
RecFeats = False
SubFeats = False

if AllFeats:
	SubsetFeatures = [False,0,N_PNs] # [Logical, Cell Start Index, Cell End Index] - Only triggers if logical = True
elif StimFeats:
	SubsetFeatures = [True,0,N_PNs_stim]
elif RecFeats:
	SubsetFeatures = [True,N_PNs_stim,N_PNs]
elif SubFeats:
	SubsetFeatures = [True,int(N_PNs_stim/2)-int(NFeats/2),int(N_PNs_stim/2)+int(NFeats/2)]

print(SubsetFeatures)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
	"""
	Annotate barplot with p-values.

	:param num1: number of left bar to put bracket over
	:param num2: number of right bar to put bracket over
	:param data: string to write or number for generating asterixes
	:param center: centers of all bars (like plt.bar() input)
	:param height: heights of all bars (like plt.bar() input)
	:param yerr: yerrs of all bars (like plt.bar() input)
	:param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
	:param barh: bar height in axes coordinates (0 to 1)
	:param fs: font size
	:param maxasterix: maximum number of asterixes to write (for very small p-values)
	"""

	if type(data) is str:
		text = data
	else:
		# * is p < 0.05
		# ** is p < 0.005
		# *** is p < 0.0005
		# etc.
		text = ''
		p = .05

		while data < p:
			text += '*'
			p /= 10.

			if maxasterix and len(text) == maxasterix:
				break

		if len(text) == 0:
			text = 'n. s.'

	lx, ly = center[num1], height[num1]
	rx, ry = center[num2], height[num2]

	if yerr:
		ly += yerr[num1]
		ry += yerr[num2]

	ax_y0, ax_y1 = plt.gca().get_ylim()
	dh *= (ax_y1 - ax_y0)
	barh *= (ax_y1 - ax_y0)

	y = max(ly, ry) + dh

	barx = [lx, lx, rx, rx]
	bary = [y, y+barh/2, y+barh/2, y]
	mid = ((lx+rx)/2, y+barh*(3/4))

	plt.plot(barx, bary, c='black')

	kwargs = dict(ha='center', va='bottom')
	if fs is not None:
		kwargs['fontsize'] = fs

	plt.text(*mid, text, **kwargs)

area_y_angle1 = np.load('output_readout/'+Metric+'y_a1.npy')
area_o_angle1 = np.load('output_readout/'+Metric+'o_a1.npy')
area_y_angle2 = np.load('output_readout/'+Metric+'y_a2.npy')
area_o_angle2 = np.load('output_readout/'+Metric+'o_a2.npy')

# inte_y_angle1 = np.load('output_readout/integy_a1.npy')
# inte_o_angle1 = np.load('output_readout/intego_a1.npy')
# inte_y_angle2 = np.load('output_readout/integy_a2.npy')
# inte_o_angle2 = np.load('output_readout/intego_a2.npy')

class_a1 = np.zeros(len(rndinds))
class_a2 = np.ones(len(rndinds))

RANDOM_STATE = 123
nbins = 10

gnb = GaussianNB(var_smoothing=10)
svc = SVC(kernel="linear", C=1.0)
lr = CalibratedClassifierCV(svc, cv=2, method='sigmoid')
mlp = MLPClassifier(hidden_layer_sizes=(700,),activation='logistic',solver='lbfgs',alpha=0.01,learning_rate='constant',learning_rate_init=0.001,max_iter=2000,random_state=123)

cmodels = [(gnb, 'Naive Bayes'),(svc, 'Support\nVector'),(mlp, 'Multi-Layer\nPerceptron')]

errors = {}
errors['y'] = {}
errors['o'] = {}
for clf, name in cmodels:
	errors['y'][name]= {}
	errors['o'][name]= {}
	errors['y'][name]['bsl'] = []
	errors['y'][name]['as'] = []
	errors['o'][name]['bsl'] = []
	errors['o'][name]['as'] = []

# Now create inputs to classifier models
for p in range(0,NPermutes):
	print('Permutation #'+str(p))
	
	# Randomly permute samples
	np.random.shuffle(area_y_angle1)
	np.random.shuffle(area_y_angle2)
	np.random.shuffle(area_o_angle1)
	np.random.shuffle(area_o_angle2)
	
	if SubsetFeatures[0]:
		tidx1 = SubsetFeatures[1]
		tidx2 = SubsetFeatures[2]
		ay1 = [temp[tidx1:tidx2] for temp in area_y_angle1]
		ay2 = [temp[tidx1:tidx2] for temp in area_y_angle2]
		ao1 = [temp[tidx1:tidx2] for temp in area_o_angle1]
		ao2 = [temp[tidx1:tidx2] for temp in area_o_angle2]
		X_train_y = np.concatenate((ay1[:num2train],ay2[:num2train]))
		X_test_y = np.concatenate((ay1[num2train:],ay2[num2train:]))
		X_train_o = np.concatenate((ao1[:num2train],ao2[:num2train]))
		X_test_o = np.concatenate((ao1[num2train:],ao2[num2train:]))
	else:
		X_train_y = np.concatenate((area_y_angle1[:num2train],area_y_angle2[:num2train]))
		X_test_y = np.concatenate((area_y_angle1[num2train:],area_y_angle2[num2train:]))
		X_train_o = np.concatenate((area_o_angle1[:num2train],area_o_angle2[:num2train]))
		X_test_o = np.concatenate((area_o_angle1[num2train:],area_o_angle2[num2train:]))
	
	y_train = np.concatenate((class_a1[:num2train],class_a2[:num2train]))
	y_test = np.concatenate((class_a1[num2train:],class_a2[num2train:]))
	
	# Young
	plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((3, 1), (2, 0))

	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
	for clf, name in cmodels:
		clf.fit(X_train_y, y_train)
		if hasattr(clf, "predict_proba"):
			prob_pos = clf.predict_proba(X_test_y)[:, 1]
		else:  # use decision function
			prob_pos = clf.decision_function(X_test_y)
			prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
		
		fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=nbins)
		
		ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
				 label="%s" % (name, ))
		
		ax2.hist(prob_pos, range=(0, 1), bins=nbins, label=name,
				 histtype="step", lw=2)
		
		pred = clf.predict(X_test_y)
		
		bsl_score = brier_score_loss(y_test, prob_pos)
		as_score = accuracy_score(y_test, pred)
		errors['y'][name]['bsl'].append(bsl_score)
		errors['y'][name]['as'].append(as_score)
	
	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')

	ax2.set_xlabel("Mean predicted value")
	ax2.set_ylabel("Count")
	# ax2.legend(loc="upper center", ncol=2)

	plt.tight_layout()
	plt.savefig('figs_calibration/classifier_y'+str(p)+'.png',dpi=300,transparent=True)
	plt.cla()   # Clear axis
	plt.clf()   # Clear figure
	plt.close() # Close a figure window
	
	# Old
	plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((3, 1), (2, 0))

	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
	for clf, name in cmodels:
		clf.fit(X_train_o, y_train)
		if hasattr(clf, "predict_proba"):
			prob_pos = clf.predict_proba(X_test_o)[:, 1]
		else:  # use decision function
			prob_pos = clf.decision_function(X_test_o)
			prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
		
		fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=nbins)
		
		ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
				 label="%s" % (name, ))
		
		ax2.hist(prob_pos, range=(0, 1), bins=nbins, label=name,
				 histtype="step", lw=2)
		
		pred = clf.predict(X_test_o)
		
		bsl_score = brier_score_loss(y_test, prob_pos)
		as_score = accuracy_score(y_test, pred)
		errors['o'][name]['bsl'].append(bsl_score)
		errors['o'][name]['as'].append(as_score)
	
	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')

	ax2.set_xlabel("Mean predicted value")
	ax2.set_ylabel("Count")
	# ax2.legend(loc="upper center", ncol=2)

	plt.tight_layout()
	plt.savefig('figs_calibration/classifier_o'+str(p)+'.png',dpi=300,transparent=True)
	plt.cla()   # Clear axis
	plt.clf()   # Clear figure
	plt.close() # Close a figure window


df = pd.DataFrame(columns=["Metric", "Classifier", "Mean Young", "SD Young", "95% CI Young", "Mean Old", "SD Old", "95% CI Old", "t-stat", "p-value","Cohen's d"])
CIasyerr = False # Use 95% CI as yerr in plots instead of SD

# Plot validation errors
labels = []
width = 0.35  # the width of the bars
x = 0
xs = []
fig, ax = plt.subplots(figsize=(21, 8))
ax.grid(axis='y',linestyle=':', linewidth=3)
ax.set_axisbelow(True)
for name,dict_ in errors['y'].items():
	labels.append(name)
	xs.append(x)
	mean_y = np.mean(errors['y'][name]['bsl'])
	mean_o = np.mean(errors['o'][name]['bsl'])
	sd_y = np.std(errors['y'][name]['bsl'])
	sd_o = np.std(errors['o'][name]['bsl'])
	CI95_y = np.percentile(errors['y'][name]['bsl'],[2.5, 97.5])
	CI95_o = np.percentile(errors['o'][name]['bsl'],[2.5, 97.5])
	tstat, pval = st.ttest_ind(errors['y'][name]['bsl'],errors['o'][name]['bsl'])
	cd = cohen_d(errors['y'][name]['bsl'],errors['o'][name]['bsl'])
	
	df = df.append({"Metric" : 'Brier Score Loss',
				"Classifier" : name,
				"Mean Young" : mean_y,
				"SD Young" : sd_y,
				"95% CI Young" : CI95_y,
				"Mean Old" : mean_o,
				"SD Old" : sd_o,
				"95% CI Old" : CI95_o,
				"t-stat" : tstat,
				"p-value" : pval,
				"Cohen's d" : cd},
				ignore_index = True)
	
	yerry = [[abs(mean_y-CI95_y[0])],[abs(mean_y-CI95_y[1])]]
	yerro = [[abs(mean_o-CI95_o[0])],[abs(mean_o-CI95_o[1])]]
	x1 = x - width*(1/2)
	x2 = x + width*(1/2)
	if x == 0:
		rects1 = ax.bar(x1, mean_y, width, yerr=yerry if CIasyerr else sd_y, label=r'Young', edgecolor='k', color='darkgray',error_kw=dict(lw=3, capsize=10, capthick=3))
		rects1 = ax.bar(x2, mean_o, width, yerr=yerro if CIasyerr else sd_o, label=r'Old', edgecolor='k', color='lightcoral',error_kw=dict(lw=3, capsize=10, capthick=3))
	else:
		rects1 = ax.bar(x1, mean_y, width, yerr=yerry if CIasyerr else sd_y, label='_nolegend_', edgecolor='k', color='darkgray',error_kw=dict(lw=3, capsize=10, capthick=3))
		rects1 = ax.bar(x2, mean_o, width, yerr=yerro if CIasyerr else sd_o, label='_nolegend_', edgecolor='k', color='lightcoral',error_kw=dict(lw=3, capsize=10, capthick=3))
	if pval < 0.001:
		barplot_annotate_brackets(0, 1, '***', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	elif pval < 0.01:
		barplot_annotate_brackets(0, 1, '**', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	elif pval < 0.05:
		barplot_annotate_brackets(0, 1, '*', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	x += 1

ax.set_ylabel('Brier Score Loss')
# ax.set_xlabel('Classifier Model')
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylim(0,1.24)
# ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('figs_readout/Scores_BrierScoreLoss_'+Metric+'_Train'+str(num2train)+'Of'+str(N_seeds)+'_Cells'+str(SubsetFeatures[1])+'to'+str(SubsetFeatures[2])+'_'+str(NPermutes)+'permutes.png',dpi=300,transparent=True)

labels = []
width = 0.40  # the width of the bars
x = 0
xs = []
fig, ax = plt.subplots(figsize=(8, 5))
# ax.grid(axis='y',linestyle=':', linewidth=3)
# ax.set_axisbelow(True)
for name,dict_ in errors['y'].items():
	labels.append(name)
	xs.append(x)
	mean_y = np.mean(errors['y'][name]['as'])
	mean_o = np.mean(errors['o'][name]['as'])
	sd_y = np.std(errors['y'][name]['as'])
	sd_o = np.std(errors['o'][name]['as'])
	CI95_y = np.percentile(errors['y'][name]['as'],[2.5, 97.5])
	CI95_o = np.percentile(errors['o'][name]['as'],[2.5, 97.5])
	tstat, pval = st.ttest_ind(errors['y'][name]['as'],errors['o'][name]['as'])
	cd = cohen_d(errors['y'][name]['as'],errors['o'][name]['as'])
	
	df = df.append({"Metric" : 'Accuracy Score',
				"Classifier" : name,
				"Mean Young" : mean_y,
				"SD Young" : sd_y,
				"95% CI Young" : CI95_y,
				"Mean Old" : mean_o,
				"SD Old" : sd_o,
				"95% CI Old" : CI95_o,
				"t-stat" : tstat,
				"p-value" : pval,
				"Cohen's d" : cd},
				ignore_index = True)
	
	yerry = [[abs(mean_y-CI95_y[0])],[abs(mean_y-CI95_y[1])]]
	yerro = [[abs(mean_o-CI95_o[0])],[abs(mean_o-CI95_o[1])]]
	x1 = x - width*(1/2)
	x2 = x + width*(1/2)
	if x == 0:
		rects1 = ax.bar(x1, mean_y, width, yerr=yerry if CIasyerr else sd_y, label=r'Young', linewidth=1, edgecolor='k', color='darkgray',error_kw=dict(lw=4, capsize=12, capthick=4))
		rects1 = ax.bar(x2, mean_o, width, yerr=yerro if CIasyerr else sd_o, label=r'Old', linewidth=1, edgecolor='k', color='lightcoral',error_kw=dict(lw=4, capsize=12, capthick=4))
	else:
		rects1 = ax.bar(x1, mean_y, width, yerr=yerry if CIasyerr else sd_y, label='_nolegend_', linewidth=1, edgecolor='k', color='darkgray',error_kw=dict(lw=4, capsize=12, capthick=4))
		rects1 = ax.bar(x2, mean_o, width, yerr=yerro if CIasyerr else sd_o, label='_nolegend_', linewidth=1, edgecolor='k', color='lightcoral',error_kw=dict(lw=4, capsize=12, capthick=4))
	if pval < 0.001:
		barplot_annotate_brackets(0, 1, '***', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	elif pval < 0.01:
		barplot_annotate_brackets(0, 1, '**', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	elif pval < 0.05:
		barplot_annotate_brackets(0, 1, '*', [x1,x2], [mean_y,mean_o], yerr=[yerry[1],yerro[1]] if CIasyerr else [sd_y,sd_o])
	x += 1

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel('Discrimination Accuracy')
# ax.set_xlabel('Classifier Model')
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylim(0.5,1.05)
# ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('figs_readout/Scores_AccuracyScore_'+Metric+'_Train'+str(num2train)+'Of'+str(N_seeds)+'_Cells'+str(SubsetFeatures[1])+'to'+str(SubsetFeatures[2])+'_'+str(NPermutes)+'permutes.png',dpi=300,transparent=True)

# Accuracy histograms
fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = True, sharey = True, figsize=(12, 15))
x = 0
for name,dict_ in errors['y'].items():
	ax[x].grid(axis='both',linestyle=':')
	ax[x].set_axisbelow(True)
	y = errors['y'][name]['as']
	o = errors['o'][name]['as']
	
	ax[x].hist(y, bins=20, range=(0,1), label=r'Young', edgecolor='k', color='darkgray',alpha=0.5)
	ax[x].hist(o, bins=20, range=(0,1), label=r'Old', edgecolor='k', color='lightcoral',alpha=0.5)
	ax[x].set_ylabel(name)
	ax[x].set_xlim(0,1)
	x += 1

ax[3].set_xlabel('Accuracy Score')
# ax[0].legend(loc='upper left')
fig.tight_layout()
fig.savefig('figs_readout/Histogram_AccuracyScore_'+Metric+'_Train'+str(num2train)+'Of'+str(N_seeds)+'_Cells'+str(SubsetFeatures[1])+'to'+str(SubsetFeatures[2])+'_'+str(NPermutes)+'permutes.png',dpi=300,transparent=True)

df.to_csv('figs_readout/stats_'+Metric+'_Train'+str(num2train)+'Of'+str(N_seeds)+'_Cells'+str(SubsetFeatures[1])+'to'+str(SubsetFeatures[2])+'_'+str(NPermutes)+'permutes.csv')
