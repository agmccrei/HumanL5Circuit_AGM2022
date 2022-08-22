COMMENT

Josh Held's adaptation to suit HCN2.  12/22/2003

****
Kinetic model of HCN2 channel gating from Wang et al 2002.

In this model channel opening is coupled to a change in the affinity of the cyclic nucleotide binding domain for cAMP which is manifest as a shift in the activation curve toward more positive potentials.  This model explains the slow activation kinetics of Ih associated with low concentrations of cAMP.

For further details email Matt Nolan at mfnolan@fido.cpmc.columbia.edu.

Reference

Wang J., Chen S., Nolan M.F. and Siegelbaum S.A. (2002). Activity-dependent regulation of HCN pacemaker channels by cyclicAMP: signalling through dynamic allosteric coupling. Neuron 36, 1-20.
****
ENDCOMMENT

: HCN2_CH, modifiable for cholinergic interneuron


NEURON {
	SUFFIX hcn2
	NONSPECIFIC_CURRENT i
	RANGE i, ehcn, g, gbar, shift
	GLOBAL a0, b0, aa0, ba0
	GLOBAL ah, bh, aah, bah
	GLOBAL ac, bc, aac, bac
	GLOBAL kon, koff, b, bf, ai, gca
}

UNITS {
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
	(S) = (siemens)
	(mA) = (milliamp)
}

PARAMETER {
	gbar    = 1		(S/cm2)
	ehcn    = -45 	    	(mV)
	a0      = .0009		(/ms)	: parameters for alpha and beta
	b0      = .0004		(/ms)
	ah      = -95		(mV)
	bh      = -51.7		(mV)
	ac      = -.12		(/mV)
	bc      = .12		(/mV)
	aa0     = 3e-05		(/ms)	: parameters for alphaa and betaa
	ba0     = .001		(/ms)
	aah     = -94.2		(mV)
	bah     = -35.5		(mV)
	aac     = -.075		(/mV)
	bac     = .144		(/mV)
	kon     = 30		(/mM-ms) : cyclic AMP binding parameters
	koff    = 4.5e-05	(/ms)
	b       = 80
	bf      = 8.94
	ai	= 1e-05		(mM)	: concentration cyclic AMP
	gca     = 1			: relative conductance of the bound state
	shift   = -12		(mV)	: shift in voltage dependence
	q10v    = 4			: q10 value from Magee 1998
	q10a    = 1.5			: estimated q10 for the cAMP binding reaction
	celsius			(degC)
}

ASSIGNED {
	v	(mV)
	g	(S/cm2)
	i	(mA/cm2)
	alpha	(/ms)
	beta    (/ms)
	alphaa	(/ms)
	betaa	(/ms)
}

STATE {
	c
	cac
	o
	cao
}

INITIAL {
	SOLVE kin STEADYSTATE sparse
}

BREAKPOINT {
	SOLVE kin METHOD sparse
	g = gbar*(o + cao*gca)
	i = g*(v-ehcn)
}

KINETIC kin {
	LOCAL qa
	qa = q10a^((celsius-22 (degC))/10 (degC))
	rates(v)
	~ c <-> o       (alpha, beta)
	~ c <-> cac     (kon*qa*ai/bf,koff*qa*b/bf)
	~ o <-> cao     (kon*qa*ai, koff*qa)
	~ cac <-> cao   (alphaa, betaa)
	CONSERVE c + cac + o + cao = 1
}

PROCEDURE rates(v(mV)) {
	LOCAL qv
	qv = q10v^((celsius-22 (degC))/10 (degC))
	if (v > -200) {
		alpha = a0*qv / (1 + exp(-(v-ah-shift)*(ac-shift)))
		beta = b0*qv / (1 + exp(-(v-bh-shift)*(bc-shift)))
		alphaa = aa0*qv / (1 + exp(-(v-aah-shift)*(aac-shift)))
		betaa = ba0*qv / (1 + exp(-(v-bah-shift)*(bac-shift)))
	} else {
		alpha = a0*qv / (1 + exp(-((-200)-ah-shift)*(ac-shift)))
		beta = b0*qv / (1 + exp(-((-200)-bh-shift)*(bc-shift)))
		alphaa = aa0*qv / (1 + exp(-((-200)-aah-shift)*(aac-shift)))
		betaa = ba0*qv / (1 + exp(-((-200)-bah-shift)*(bac-shift)))
	}
}
