import numpy as np

from classification import EvaluateResults
from classification import KFA
from classification import NNClassify
from classification import Projection


def compute_avg_EER(fae, fre):
	C = fae + fre;
	maxi1 = float('inf')
	resolu = len(fae);
	for i in range(0, resolu):

		if abs(fae[i] - fre[i]) < maxi1:
			index1 = i;
			maxi1 = abs(fae[i] - fre[i]);

	ERR_value = C[index1] / 2 * 100
	# print(index1)
	return ERR_value


def average_output(Avg_far, Avg_frr, Avg_rec_rates, Nfolds):
	FAR = np.zeros(len(Avg_far[0]))
	FRR = np.zeros(len(Avg_frr[0]))

	Rec_Rates = np.zeros(len(Avg_rec_rates[0]))

	for i in range(0, len(Avg_far)):
		FAR += Avg_far[i]
		FRR += Avg_frr[i]
		Rec_Rates += Avg_rec_rates[i]

	FAR = FAR / Nfolds
	FRR = FRR / Nfolds
	Rec_Rates = Rec_Rates / Nfolds
	return FAR, FRR, Rec_Rates


def compute_results(train_data, ids_train, test_data, ids_test):
	model = KFA.perform_kfa_PhD(train_data, ids_train, 'fpp', len(ids_test) - 1)

	print('Finished KFA subspace construction. Starting evaluation and test image projection.')
	test_features = Projection.nonlinear_subspace_projection_PhD(test_data, model)

	results = NNClassify.nn_classification_PhD(model.train, ids_train, test_features, ids_test, test_features.shape[0],
	                                           'cos');
	output = EvaluateResults.evaluate_results_PhD(results, 'image');
	# DI=CalculateDI(results);
	# output.DI=DI;

	return model, test_features, results, output
