import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def r2_standardiser(raw_predictions, library_xs):
	"""Both arguments must be lists.
	Function returns the r2 calculated from the threshold onwards"""
	gated_predictions = []
	for xs_raw in raw_predictions:
		if xs_raw >= 0.003: # arbitrary threshold
			gated_predictions.append(xs_raw)
		else:
			gated_predictions.append(0)

	threshold_gated_predictions = []
	truncated_library_xs = []
	for i, XS in enumerate(gated_predictions):
		if XS > 0.0:
			threshold_gated_predictions.append(XS)
			truncated_library_xs.append(library_xs[i])

	standardised_r2 = r2_score(truncated_library_xs, threshold_gated_predictions)

	return(threshold_gated_predictions, truncated_library_xs, standardised_r2)

def General_plotter(df, nuclides):
	"""df: dataframe source of XSs
	nuclides: must be array of 1x2 arrays [z,a]

	Returns XS and ERG values. Designed for plotting graphs and doing r2 comparisons without running make_test
	which is much more demanding"""

	ztest = [nuclide[0] for nuclide in nuclides]  # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]

	Z = df['Z']
	A = df['A']

	Energy = df['ERG']
	XS = df['XS']

	Z_test = []
	A_test = []
	Energy_test = []
	XS_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] <= 20:
				Z_test.append(Z[j])
				A_test.append(A[j])
				Energy_test.append(Energy[j])
				XS_test.append(XS[j])

	energies = np.array(Energy_test)

	xs = XS_test

	return energies, xs




def dsigma_dE(XS):
    dsigma = [x - y for x, y in zip(XS[1:], XS[:-1])]
    n_turning_points = sum(dsig1 * dsig2 < 0 for dsig1, dsig2 in zip(dsigma[1:], dsigma[:-1]))
    # if there is a turning point, then consecutive gradients will have opposite polarities.
    # Hence their product will be negative, which counts towards the number of turning points in
    # the function.
    return(n_turning_points)




def feature_fetcher(feature, df, z, a):
	"""feature: String of feature name (corresponding to column in dataframe)
	df: dataframe, Library of choice (not that it matters, but TENDL is the obvious
	choice as it's the most complete.
	passed_nuclide: in format [z,a]"""

	truncated_df = df[df.Z == z]
	truncated_df = truncated_df[truncated_df.A == a]
	truncated_df.index = range(len(truncated_df))

	retrieved_feature = truncated_df[feature][1]

	return(retrieved_feature)




def anomaly_remover(dfa):
	anomalies = [[14, 28], [14, 29], [14, 30], [15, 31], [20, 40], [20, 42], [20, 43], [20, 44], [20, 46],
				 [20, 48], [24, 50], [24, 52], [24, 53], [24, 54], [28, 58], [28, 60], [28, 61], [28, 62], [28, 64],
				 [29, 63], [29, 65], [41, 93], [80, 196], [80, 198], [80, 199], [80, 200], [80, 201], [80, 202],
				 [80, 204], [82, 206], [82, 207]]
	"""list of nuclides with cutoff value @ 20 MeV - format is [Z,A]"""

	anomaly_indices = []
	for set in anomalies:
		z = set[0]
		a = set[1]

		remover = dfa[dfa.A == a]
		remove = remover[remover.Z == z]

		last_row_index = remove.index[-1]
		anomaly_indices.append(last_row_index)


	for i in anomaly_indices:
		dfa = dfa.drop(index=i)

	dfa.index = range(len(dfa))

	return dfa

def range_setter(df, la, ua):
	"""takes dataframe as input, with lower and upper bounds of A for nuclides desired. Returns a single array,
	containing 1x2 arrays which contain nuclides in the format [Z,A]."""
	nucs = []

	for i, j, in zip(df['A'], df['Z']):  # i is A, j is Z
		if [j, i] in nucs or i > ua or i < la:
			continue
		else:
			nucs.append([j, i])  # format is [Z, A]
	return nucs

def interpolate_pandas(df, inter_col):
	"""Performs linear interpolation on XS data, using pandas.
	df: dataframe to be interpolated
	inter_col: columns of data to be interpolated. In this case, only XS and ERG

	returns interpolated dataframe"""

	all_cols = df.columns.to_list() # list of header names (strings)
	remain_cols = [c for c in all_cols if c not in inter_col] # list of columns which will NOT be interpolated

	df_inter = pd.DataFrame(columns=df.columns) # empty dataframe with column names only

	for i, row in df.iterrows():
		df_inter = df_inter._append(row, ignore_index=True)

		# Pandas interpolation works by filling out NaN values. We must create alternating rows with NaN values for XS and ERG,
		# then fill them out later
		df_inter = df_inter._append(pd.Series(np.nan, index=df.columns), ignore_index=True)

	for c in inter_col:
		df_inter[c] = df_inter[c].interpolate() # performs linear interpolation

	df_inter[remain_cols] = df_inter[remain_cols].ffill()

	# Pandas interpolates values for XS and ERG between adjacent nuclides, so the last element for each nuclide must be removed

	final_index = []
	completed_removal = []
	for z, a in zip(df_inter['Z'], df_inter['A']):
		if [z,a] in completed_removal:
			continue

		# form dataframe containing isolated nuclide, then save index of last element for use in .drop()
		remover = df_inter[df_inter.A == a]
		remove = remover[remover.Z == z]

		last_row_index = remove.index[-1]
		final_index.append(last_row_index)
		completed_removal.append([z,a])
	for i in final_index:
		df_inter = df_inter.drop(index=i) # removes value from whole dataframe

	df_inter.index = range(len(df_inter)) # re-index dataframe
	return df_inter

def make_train(df, validation_nuclides, la=0, ua=260):
	"""la: lower bound for A
	ua: upper bound for A
	arguments la and ua allow data stratification using A
	df: dataframe to use
	validation_nuclides are explicitly omitted from the training matrix
	Returns X: X values matrix in shape (nsamples, nfeatures)"""

	# MT = df['MT']
	ME = df['ME']
	Z = df['Z']
	A = df['A']
	Sep_2n = df['S2n']
	Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	Sep_p = df['Sp']
	Sep_n = df['Sn']
	BEA = df['BEA']
	# Pairing = df['Pairing']
	gamma_deformation = df['gamma_deformation']
	beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']
	N = df['N']
	# xs_max = df['xs_max']
	Radius = df['Radius']
	Shell = df['Shell']
	Parity = df['Parity']
	Spin = df['Spin']
	Decay_Const = df['Decay_Const']
	Deform = df['Deform']
	n_gap_erg = df['n_gap_erg']
	n_chem_erg = df['n_chem_erg']
	p_gap_erg = df['p_gap_erg']
	p_chem_erg = df['p_chem_erg']
	n_rms_radius = df['n_rms_radius']
	p_rms_radius = df['p_rms_radius']
	# rms_radius = df['rms_radius']
	# magic_p = df['cat_magic_proton']
	# magic_n = df['cat_magic_neutron']
	# magic_d = df['cat_magic_double']
	Nlow = df['Nlow']
	Ulow = df['Ulow']
	# Ntop = df['Ntop']
	Utop = df['Utop']
	ainf = df['ainf']
	# XSlow = df['XSlow']
	# XSupp = df['XSupp']
	Asymmetry = df['Asymmetry']

	# Compound nucleus properties
	Sp_compound = df['Sp_compound']
	Sn_compound = df['Sn_compound']
	S2n_compound = df['S2n_compound']
	S2p_compound = df['S2p_compound']
	Decay_compound = df['Decay_compound']
	BEA_compound = df['BEA_compound']
	BEA_A_compound = df['BEA_A_compound']
	Radius_compound = df['Radius_compound']
	# ME_compound = df['ME_compound']
	# Pairing_compound = df['Pairing_compound']
	Shell_compound = df['Shell_compound']
	Spin_compound = df['Spin_compound']
	# Parity_compound = df['Parity_compound']
	Deform_compound = df['Deform_compound']
	Asymmetry_compound = df['Asymmetry_compound']

	# Daughter nucleus properties
	Sn_daughter = df['Sn_daughter']
	S2n_daughter = df['S2n_daughter']
	Sp_daughter = df['Sp_daughter']
	# Pairing_daughter = df['Pairing_daughter']
	# Parity_daughter = df['Parity_daughter']
	BEA_daughter = df['BEA_daughter']
	# Shell_daughter = df['Shell_daughter']
	# S2p_daughter = df['S2p_daughter']
	# Radius_daughter = df['Radius_daughter']
	ME_daughter = df['ME_daughter']
	BEA_A_daughter = df['BEA_A_daughter']
	# Spin_daughter = df['Spin_daughter']
	Deform_daughter = df['Deform_daughter']
	Decay_daughter = df['Decay_daughter']
	Asymmetry_daughter = df['Asymmetry_daughter']

	# AM = df['AM']

	Z_train = []
	A_train = []
	S2n_train = []
	S2p_train = []
	Energy_train = []
	XS_train = []
	Sp_train = []
	Sn_train = []
	BEA_train = []
	# Pairing_train = []
	gd_train = []
	N_train = [] # neutron number
	bd_train = []
	Radius_train = []
	n_gap_erg_train = []
	n_chem_erg_train = []
	# xs_max_train = []
	n_rms_radius_train = []
	# octupole_deformation_train = []
	# cat_proton_train = []
	# cat_neutron_train = []
	# cat_double_train = []
	ME_train = []
	# Z_even_train = []
	# A_even_train = []
	# N_even_train = []
	Shell_train = []
	Parity_train = []
	Spin_train = []
	Decay_Const_train = []
	Deform_train = []
	p_gap_erg_train = []
	p_chem_erg_train = []
	p_rms_radius_train = []
	# rms_radius_train = []
	Nlow_train = []
	Ulow_train = []
	# Ntop_train = []
	Utop_train = []
	ainf_train = []
	# XSlow_train = []
	# XSupp_train = []
	Asymmetry_train = []



	# Daughter nucleus properties
	Sn_d_train = []
	Sp_d_train = []
	S2n_d_train = []
	# Pairing_daughter_train = []
	# Parity_daughter_train = []
	BEA_daughter_train = []
	# S2p_daughter_train = []
	# Shell_daughter_train = []
	Decay_daughter_train = []
	ME_daughter_train = []
	# Radius_daughter_train = []
	BEA_A_daughter_train = []
	# Spin_daughter_train = []
	Deform_daughter_train = []
	Asymmetry_daughter_train = []

	# Compound nucleus properties
	Sn_c_train = []  # Sn compound
	Sp_compound_train = []
	BEA_compound_train = []
	S2n_compound_train = []
	S2p_compound_train = []
	Decay_compound_train = []
	# Sn_compound_train = []
	Shell_compound_train = []
	Spin_compound_train = []
	Radius_compound_train = []
	Deform_compound_train = []
	ME_compound_train = []
	BEA_A_compound_train = []
	# Pairing_compound_train = []
	# Parity_compound_train = []
	Asymmetry_compound_train = []

	# AM_train = []

	for idx, unused in enumerate(Z):  # MT = 16 is (n,2n) (already extracted)
		if [Z[idx], A[idx]] in validation_nuclides:
			continue # prevents loop from adding test isotope data to training data
		if Energy[idx] >= 21: # training on data less than 30 MeV
			continue
		if A[idx] <= ua and A[idx] >= la: # checks that nuclide is within bounds for A
			Z_train.append(Z[idx])
			A_train.append(A[idx])
			S2n_train.append(Sep_2n[idx])
			S2p_train.append(Sep_2p[idx])
			Energy_train.append(Energy[idx])
			XS_train.append(XS[idx])
			Sp_train.append(Sep_p[idx])
			Sn_train.append(Sep_n[idx])
			BEA_train.append(BEA[idx])
			# Pairing_train.append(Pairing[idx])
			Sn_c_train.append(Sn_compound[idx])
			gd_train.append(gamma_deformation[idx])
			N_train.append(N[idx])
			bd_train.append(beta_deformation[idx])
			Sn_d_train.append(Sn_daughter[idx])
			Sp_d_train.append(Sp_daughter[idx])
			S2n_d_train.append(S2n_daughter[idx])
			Radius_train.append(Radius[idx])
			n_gap_erg_train.append(n_gap_erg[idx])
			n_chem_erg_train.append(n_chem_erg[idx])
			# Pairing_daughter_train.append(Pairing_daughter[idx])
			# Parity_daughter_train.append(Parity_daughter[idx])
			# xs_max_train.append(xs_max[idx])
			n_rms_radius_train.append(n_rms_radius[idx])
			# octupole_deformation_train.append(octupole_deformation[idx])
			Decay_compound_train.append(Decay_compound[idx])
			BEA_daughter_train.append(BEA_daughter[idx])
			BEA_compound_train.append(BEA_compound[idx])
			S2n_compound_train.append(S2n_compound[idx])
			S2p_compound_train.append(S2p_compound[idx])
			ME_train.append(ME[idx])
			# Z_even_train.append(Z_even[idx])
			# A_even_train.append(A_even[idx])
			# N_even_train.append(N_even[idx])
			Shell_train.append(Shell[idx])
			Parity_train.append(Parity[idx])
			Spin_train.append(Spin[idx])
			Decay_Const_train.append(Decay_Const[idx])
			Deform_train.append(Deform[idx])
			p_gap_erg_train.append(p_gap_erg[idx])
			p_chem_erg_train.append(p_chem_erg[idx])
			p_rms_radius_train.append(p_rms_radius[idx])
			# rms_radius_train.append(rms_radius[idx])
			Sp_compound_train.append(Sp_compound[idx])
			# Sn_compound_train.append(Sn_compound[idx])
			Shell_compound_train.append(Shell_compound[idx])
			# S2p_daughter_train.append(S2p_daughter[idx])
			# Shell_daughter_train.append(Shell_daughter[idx])
			Spin_compound_train.append(Spin_compound[idx])
			Radius_compound_train.append(Radius_compound[idx])
			Deform_compound_train.append(Deform_compound[idx])
			# ME_compound_train.append(ME_compound[idx])
			BEA_A_compound_train.append(BEA_A_compound[idx])
			Decay_daughter_train.append(Decay_daughter[idx])
			ME_daughter_train.append(ME_daughter[idx])
			# Radius_daughter_train.append(Radius_daughter[idx])
			# Pairing_compound_train.append(Pairing_compound[idx])
			# Parity_compound_train.append(Parity_compound[idx])
			BEA_A_daughter_train.append(BEA_A_daughter[idx])
			# Spin_daughter_train.append(Spin_daughter[idx])
			Deform_daughter_train.append(Deform_daughter[idx])
			Nlow_train.append(Nlow[idx])
			Ulow_train.append(Ulow[idx])
			# Ntop_train.append(Ntop[idx])
			Utop_train.append(Utop[idx])
			ainf_train.append(ainf[idx])
			# XSlow_train.append(XSlow[idx])
			# XSupp_train.append(XSupp[idx])
			Asymmetry_train.append(Asymmetry[idx])
			Asymmetry_compound_train.append(Asymmetry_compound[idx])
			Asymmetry_daughter_train.append(Asymmetry_daughter[idx])
			# AM_train.append(AM[idx])

	X = np.array([Z_train,
				  A_train,
				  S2n_train,
				  S2p_train,
				  Energy_train,
				  Sp_train,
				  Sn_train,
				  BEA_train,
				  # Pairing_train,
				  Sn_c_train,
				  gd_train,
				  N_train,
				  bd_train,
				  Sn_d_train,
				  # Sp_d_train,
				  S2n_d_train,
				  Radius_train,
				  n_gap_erg_train,
				  n_chem_erg_train,
				  # xs_max_train,
				  n_rms_radius_train,
				  # octupole_deformation_train,
				  # Decay_compound_train,
				  BEA_daughter_train,
				  BEA_compound_train,
				  # Pairing_daughter_train,
				  # Parity_daughter_train,
				  # S2n_compound_train,
				  S2p_compound_train,
				  ME_train,
				  # Z_even_train,
				  # A_even_train,
				  # N_even_train,
				  Shell_train,
				  # Parity_train,
				  # Spin_train,
				  Decay_Const_train,
				  # Deform_train,
				  p_gap_erg_train,
				  p_chem_erg_train,
				  # p_rms_radius_train,
				  # rms_radius_train,
				  # Sp_compound_train,
				  # Sn_compound_train,
				  Shell_compound_train,
				  # S2p_daughter_train,
				  # Shell_daughter_train,
				  Spin_compound_train,
				  # Radius_compound_train,
				  Deform_compound_train,
				  # ME_compound_train,
				  BEA_A_compound_train,
				  Decay_daughter_train,
				  # ME_daughter_train,
				  # Radius_daughter_train,
				  # Pairing_compound_train,
				  # Parity_compound_train,
				  BEA_A_daughter_train,
				  # Spin_daughter_train,
				  Deform_daughter_train,
				  # cat_proton_train,
				  # cat_neutron_train,
				  # cat_double_train,
				  Nlow_train,
				  # Ulow_train,
				  # Ntop_train,
				  Utop_train,
				  # ainf_train,
				  # XSlow_train,
				  # XSupp_train,
				  Asymmetry_train,
				  # Asymmetry_compound_train,
				  Asymmetry_daughter_train,
				  # AM_train,
				  ])
	y = np.array(XS_train) # cross sections

	X = np.transpose(X) # forms matrix into correct shape (values, features)
	return X, y

def make_test(nuclides, df):
	"""
	nuclides: array of (1x2) arrays containing Z of nuclide at index 0, and A at index 1.
	df: dataframe used for validation data

	Returns: xtest - matrix of values for the model to use for making predictions
	ytest: cross sections
	"""

	ztest = [nuclide[0] for nuclide in nuclides] # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]


	# MT = df['MT']
	# AM = df['AM']
	ME = df['ME']
	Z = df['Z']
	A = df['A']
	S_2n = df['S2n']
	S_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	S_p = df['Sp']
	S_n = df['Sn']
	BEA = df['BEA']
	# Pairing = df['Pairing']
	gamma_deformation = df['gamma_deformation']
	beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']
	N = df['N']
	# xs_max = df['xs_max']
	Radius = df['Radius']
	Shell = df['Shell']
	Parity = df['Parity']
	Spin = df['Spin']
	Decay_Const = df['Decay_Const']
	Deform = df['Deform']
	n_gap_erg = df['n_gap_erg']
	n_chem_erg = df['n_chem_erg']
	p_gap_erg = df['p_gap_erg']
	p_chem_erg = df['p_chem_erg']
	n_rms_radius = df['n_rms_radius']
	p_rms_radius = df['p_rms_radius']
	# rms_radius = df['rms_radius']
	# magic_p = df['cat_magic_proton']
	# magic_n = df['cat_magic_neutron']
	# magic_d = df['cat_magic_double']
	Nlow = df['Nlow']
	Ulow = df['Ulow']
	# Ntop = df['Ntop']
	Utop = df['Utop']
	# ainf = df['ainf']
	# XSlow = df['XSlow']
	# XSupp = df['XSupp']
	Asymmetry = df['Asymmetry']

	# Compound nucleus properties
	Sp_compound = df['Sp_compound']
	Sn_compound = df['Sn_compound']
	S2n_compound = df['S2n_compound']
	S2p_compound = df['S2p_compound']
	Decay_compound = df['Decay_compound']
	BEA_compound = df['BEA_compound']
	BEA_A_compound = df['BEA_A_compound']
	Radius_compound = df['Radius_compound']
	# ME_compound = df['ME_compound']
	# Pairing_compound = df['Pairing_compound']
	Shell_compound = df['Shell_compound']
	Spin_compound = df['Spin_compound']
	# Parity_compound = df['Parity_compound']
	Deform_compound = df['Deform_compound']
	Asymmetry_compound = df['Asymmetry_compound']

	# Daughter nucleus properties
	Sn_daughter = df['Sn_daughter']
	S2n_daughter = df['S2n_daughter']
	Sp_daughter = df['Sp_daughter']
	# Pairing_daughter = df['Pairing_daughter']
	# Parity_daughter = df['Parity_daughter']
	BEA_daughter = df['BEA_daughter']
	# Shell_daughter = df['Shell_daughter']
	# S2p_daughter = df['S2p_daughter']
	# Radius_daughter = df['Radius_daughter']
	ME_daughter = df['ME_daughter']
	BEA_A_daughter = df['BEA_A_daughter']
	# Spin_daughter = df['Spin_daughter']
	Deform_daughter = df['Deform_daughter']
	Decay_daughter = df['Decay_daughter']
	Asymmetry_daughter = df['Asymmetry_daughter']

	# AM_test = []
	Z_test = []
	A_test = []
	S2n_test = []
	S2p_test = []
	Energy_test = []
	XS_test = []
	Sp_test = []
	Sn_test = []
	BEA_test = []
	# Pairing_test = []
	Sn_c_test = []
	gd_test = [] # gamma deformation
	N_test = []
	bd_test = []
	Radius_test = []
	n_gap_erg_test = []
	n_chem_erg_test = []
	ME_test = []
	# Z_even_test = []
	# A_even_test = []
	# N_even_test = []
	Shell_test = []
	Parity_test = []
	Spin_test = []
	Decay_Const_test = []
	Deform_test = []
	p_gap_erg_test = []
	p_chem_erg_test = []
	p_rms_radius_test = []
	# rms_radius_test = []
	# xs_max_test = []
	# octupole_deformation_test = []
	Nlow_test = []
	Ulow_test = []
	# Ntop_test = []
	Utop_test = []
	# ainf_test = []
	# XSlow_test = []
	# XSupp_test = []
	Asymmetry_test = []


	# Daughter features
	Sn_d_test = []
	Sp_d_test = []
	S2n_d_test = []
	n_rms_radius_test = []
	Decay_compound_test = []
	# S2p_daughter_test = []
	# Shell_daughter_test = []
	Decay_daughter_test = []
	ME_daughter_test = []
	BEA_daughter_test = []
	# Radius_daughter_test = []
	BEA_A_daughter_test = []
	# Spin_daughter_test = []
	Deform_daughter_test = []
	# Pairing_daughter_test = []
	# Parity_daughter_test = []
	Asymmetry_daughter_test = []

	Sp_compound_test = []
	# Sn_compound_test = []
	Shell_compound_test = []
	Spin_compound_test = []
	Radius_compound_test  = []
	Deform_compound_test = []
	ME_compound_test = []
	BEA_A_compound_test = []
	# Pairing_compound_test = []
	# Parity_compound_test = []
	BEA_compound_test = []
	S2n_compound_test = []
	S2p_compound_test = []
	Asymmetry_compound_test = []

	# cat_proton_test = []
	# cat_neutron_test = []
	# cat_double_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] <= 20:
				Z_test.append(Z[j])
				A_test.append(A[j])
				S2n_test.append(S_2n[j])
				S2p_test.append(S_2p[j])
				Energy_test.append(Energy[j])
				XS_test.append(XS[j])
				Sp_test.append(S_p[j])
				Sn_test.append(S_n[j])
				BEA_test.append(BEA[j])
				# Pairing_test.append(Pairing[j])
				Sn_c_test.append(Sn_compound[j])
				gd_test.append(gamma_deformation[j])
				N_test.append(N[j])
				bd_test.append(beta_deformation[j])
				Sn_d_test.append(Sn_daughter[j])
				# Sp_d_test.append(Sp_daughter[j])
				S2n_d_test.append(S2n_daughter[j])
				Radius_test.append(Radius[j])
				n_gap_erg_test.append(n_gap_erg[j])
				n_chem_erg_test.append(n_chem_erg[j])
				# Pairing_daughter_test.append(Pairing_daughter[j])
				# xs_max_test.append(np.nan) # cheat feature - nan
				# octupole_deformation_test.append(octupole_deformation[j])

				# Parity_daughter_test.append(Parity_daughter[j])
				n_rms_radius_test.append(n_rms_radius[j])
				Decay_compound_test.append(Decay_compound[j]) # D_c
				BEA_daughter_test.append(BEA_daughter[j])

				BEA_compound_test.append(BEA_compound[j])

				S2n_compound_test.append(S2n_compound[j])
				S2p_compound_test.append(S2p_compound[j])

				ME_test.append(ME[j])
				# Z_even_test.append(Z_even[j])
				# A_even_test.append(A_even[j])
				# N_even_test.append(N_even[j])
				Shell_test.append(Shell[j])
				Parity_test.append(Parity[j])
				Spin_test.append(Spin[j])
				Decay_Const_test.append(Decay_Const[j])
				Deform_test.append(Deform[j])
				p_gap_erg_test.append(p_gap_erg[j])
				p_chem_erg_test.append(p_chem_erg[j])
				p_rms_radius_test.append(p_rms_radius[j])
				# rms_radius_test.append(rms_radius[j])
				Sp_compound_test.append(Sp_compound[j])
				# Sn_compound_test.append(Sn_compound[j])
				Shell_compound_test.append(Shell_compound[j])
				# S2p_daughter_test.append(S2p_daughter[j])
				# Shell_daughter_test.append(Shell_daughter[j])
				Spin_compound_test.append(Spin_compound[j])
				Radius_compound_test.append(Radius_compound[j])
				Deform_compound_test.append(Deform_compound[j])
				# ME_compound_test.append(ME_compound[j])
				BEA_A_compound_test.append(BEA_A_compound[j])
				Decay_daughter_test.append(Decay_daughter[j])
				ME_daughter_test.append(ME_daughter[j])
				# Radius_daughter_test.append(Radius_daughter[j])
				# AM_test.append(AM[j])
				# Pairing_compound_test.append(Pairing_compound[j])
				# Parity_compound_test.append(Parity_compound[j])
				BEA_A_daughter_test.append(BEA_A_daughter[j])
				# Spin_daughter_test.append(Spin_daughter[j])
				Deform_daughter_test.append(Deform_daughter[j])
				Nlow_test.append(Nlow[j])
				Ulow_test.append(Ulow[j])
				# Ntop_test.append(Ntop[j])
				Utop_test.append(Utop[j])
				# ainf_test.append(ainf[j])
				# XSlow_test.append(XSlow[j])
				# XSupp_test.append(XSupp[j])
				Asymmetry_test.append(Asymmetry[j])
				Asymmetry_compound_test.append(Asymmetry_compound[j])
				Asymmetry_daughter_test.append(Asymmetry_daughter[j])


	xtest = np.array([Z_test,
	A_test,
	S2n_test,
	S2p_test,
	Energy_test,
	Sp_test,
	Sn_test,
	BEA_test,
	# Pairing_train,
	Sn_c_test,
	gd_test,
	N_test,
	bd_test,
	Sn_d_test,
	# Sp_d_train,
	S2n_d_test,
	Radius_test,
	n_gap_erg_test,
	n_chem_erg_test,
	# xs_max_train,
	n_rms_radius_test,
	# octupole_deformation_train,
	# Decay_compound_train,
	BEA_daughter_test,
	  BEA_compound_test,
	# Pairing_daughter_train,
	# Parity_daughter_train,
	# S2n_compound_train,
	S2p_compound_test,
	ME_test,
	# Z_even_train,
	# A_even_train,
	# N_even_train,
	Shell_test,
	# Parity_test,
	# Spin_test,
	Decay_Const_test,
	# Deform_train,
	p_gap_erg_test,
	p_chem_erg_test,
	# p_rms_radius_train,
	# rms_radius_train,
	# Sp_compound_test,
	# Sn_compound_train,
	Shell_compound_test,
	# S2p_daughter_train,
	# Shell_daughter_train,
	  Spin_compound_test,
	# Radius_compound_train,
	Deform_compound_test,
	# ME_compound_train,
	BEA_A_compound_test,
	Decay_daughter_test,
	# ME_daughter_train,
	# Radius_daughter_train,
	# Pairing_compound_train,
	# Parity_compound_train,
	BEA_A_daughter_test,
	# Spin_daughter_train,
	Deform_daughter_test,
	# cat_proton_train,
	# cat_neutron_train,
	# cat_double_train,
	Nlow_test,
	  # Ulow_train,
	# Ntop_train,
	Utop_test,
	# ainf_train,
	# XSlow_train,
	Asymmetry_test,
	# Asymmetry_compound_train,
	Asymmetry_daughter_test,
	# AM_test,
	])

	xtest = np.transpose(xtest)

	y_test = XS_test

	return xtest, y_test



