import numpy as np


def Compute_FAR_FRR(true_scores, false_scores):
	true_scores = np.array(true_scores)
	true_sorted = true_scores.flatten()
	true_sorted.sort()

	num_true = len(true_sorted)  # 360
	true_sorted = list(true_sorted)
	true_sorted.append(50 ** 6)
	true_sorted = np.array(true_sorted)

	# print('t',len(true_sorted))

	false_scores = np.array(false_scores)
	false_sorted = false_scores.flatten()
	false_sorted.sort()
	false_sorted = list(false_sorted)
	false_sorted.append(50 ** 6)
	false_sorted = np.array(false_sorted)
	num_false = len(false_sorted)
	# print('f',len(false_sorted))

	Pfrr = np.zeros(num_true + num_false + 1)
	Pfar = np.zeros(num_true + num_false + 1)

	npts = 0;  # initialize index

	Pfrr[npts] = 0.0;
	Pfar[npts] = 1.0;

	ntrue = 0;  # initialize ntrue
	nfalse = 0;  # initialize nfalse

	while ntrue < num_true or nfalse < num_false:
		if ntrue != num_true and nfalse != num_true and true_sorted[ntrue] <= false_sorted[nfalse]:
			ntrue = ntrue + 1
		# print('true')
		# print(ntrue)
		else:
			nfalse = nfalse + 1
		# print('nfalse')
		# print(nfalse)

		npts = npts + 1

		Pfrr[npts] = (ntrue - 1) / num_true;

		'''if Pfrr[npts]<0:
			Pfrr[npts]=0
		'''

		Pfar[npts] = (num_false - (nfalse - 1)) / num_false;

	return Pfrr, Pfar


def Compute_DET(true_scores, false_scores):
	# function [Pmiss, Pfa] = Compute_DET (true_scores, false_scores)
	#
	#  Compute_DET computes the (observed) miss/false_alarm probabilities
	#  for a set of detection output scores.
	#
	#  true_scores (false_scores) are detection output scores for a set of
	#  detection trials, given that the target hypothesis is true (false).
	#          (By convention, the more positive the score,
	#          the more likely is the target hypothesis.)
	#
	#  Pdet is a two-column matrix containing the detection probability
	#  trade-off.  The first column contains the miss probabilities and
	#  the second column contains the corresponding false alarm
	#  probabilities.
	#
	#  See DET_usage for examples on how to use this function.

	SMAX = float('inf');

	# this code is matlab-tized for speed.
	# speedup: Old routine 54 secs -> new routine 5.71 secs.
	# for 109776 points.

	# -------------------------
	# Compute the miss/false_alarm error probabilities

	# print('det')

	num_true = max((true_scores).shape);
	# print('numtrue',(true_scores).shape)
	num_false = max((false_scores).shape);

	total = num_true + num_false;

	Pmiss = np.zeros((num_true + num_false + 1));  # preallocate for speed
	Pfa = np.zeros((num_true + num_false + 1));  # preallocate for speed

	false_sorted = (false_scores)
	true_sorted = (true_scores)

	scores = []
	for i in range(0, total):
		scores.append([])
		for j in range(0, 2):
			scores[i].append(0)
			if j == 0:
				if (i < num_false):
					scores[i][j] = false_sorted[0][i]
				else:
					scores[i][j] = true_sorted[0][num_false - i]
			else:
				if j == 1:
					if (i < num_false):
						scores[i][j] = 0
					else:
						scores[i][j] = 1

	# print(true_sorted)
	scores = np.array(scores)
	# scores=scores.reshape(400,2)
	# print(scores)

	scores = scores[scores[:, 0].argsort()]
	'''for i in range(0,len(scores)):
		print(scores[i])
	'''
	# scores=np.sort(scores,axis=0)
	# print(scores)
	sumtrue = np.cumsum(scores[:, 1], axis=0)  ###########Doubt
	# print(sumtrue)
	sumfalse = num_false - ((np.arange(1, total + 1)).T - sumtrue)
	# print(sumfalse)
	Pmiss[0] = 0;
	Pfa[0] = 1.0;
	for i in range(0, len(sumtrue)):
		Pmiss[i + 1] = sumtrue[i] / num_true

	for i in range(0, len(sumfalse)):
		Pfa[i + 1] = sumfalse[i] / num_false

	# Pmiss[1:total+1] = sumtrue  / num_true;
	# Pfa[1:total+1]  = sumfalse / num_false;

	return Pmiss, Pfa


'''
def DETsort(x,col=None):
    # DETsort Sort rows, the first in ascending, the remaining in decending
    # thereby postponing the false alarms on like scores.
    # based on SORTROWS

    if nargin<1, error('Not enough input arguments.'); end
    if ndims(x)>2, error('X must be a 2-D matrix.'); end


    if nargin<2, col = 1:size(x,2); end
    if isempty(x), y = x; ndx = []; return, end

    [dum,S]=x.shape
    col = np.arange(1,S+1)

    ndx = (np.arange(0,len(x))).T;

    # sort 2nd column ascending
    [v,ind] = sort(x[ndx,2]);  Cannot be done
    ndx = ndx(ind);

    # reverse to decending order
    ndx(1:size(x,1)) = ndx(size(x,1):-1:1);

    # now sort first column ascending
    [v,ind] = sort(x(ndx,1));
    ndx = ndx(ind);
    y = x(ndx,:);

    return y,ndx
'''
# old routine for reference  Old_Compute_DET

'''
def Compute_DET (true_scores, false_scores):
    #function [Pmiss, Pfa] = Compute_DET (true_scores, false_scores)
    #
    #  Compute_DET computes the (observed) miss/false_alarm probabilities
    #  for a set of detection output scores.
    #
    #  true_scores (false_scores) are detection output scores for a set of
    #  detection trials, given that the target hypothesis is true (false).
    #          (By convention, the more positive the score,
    #          the more likely is the target hypothesis.)
    #
    #  Pdet is a two-column matrix containing the detection probability
    #  trade-off.  The first column contains the miss probabilities and
    #  the second column contains the corresponding false alarm
    #  probabilities.
    #
    #  See DET_usage for examples on how to use this function.

    SMAX = float('inf');

    #-------------------------
    #Compute the miss/false_alarm error probabilities

    num_true = max(true_scores.shape);
    true_sorted = np.sort(true_scores);

    true_sorted[num_true] = SMAX;

    num_false = max(false_scores.shape);
    false_sorted = np.sort(false_scores);

    false_sorted(num_false+1) = SMAX;

    #Pdet = zeros(num_true+num_false+1, 2); #preallocate Pdet for speed
    Pmiss = np.zeros((num_true+num_false+1, 1)); #preallocate for speed
    Pfa   = np.zeros((num_true+num_false+1, 1)); #preallocate for speed

    npts = 1;
    #Pdet(npts, 1:2) = [0.0 1.0];
    Pmiss(npts) = 0.0;
    Pfa(npts) = 1.0;
    ntrue = 1;
    nfalse = 1;
    num_true
    num_false
    while ntrue <= num_true | nfalse <= num_false
            if true_sorted(ntrue) <= false_sorted(nfalse)
                    ntrue = ntrue+1;
            else
                    nfalse = nfalse+1;
            end
            npts = npts+1;
    #        Pdet(npts, 1:2) = [             (ntrue-1)   / num_true ...
    #                           (num_false - (nfalse-1)) / num_false];
            Pmiss(npts) =              (ntrue-1)   / num_true;
            Pfa(npts)   = (num_false - (nfalse-1)) / num_false;
    [npts ntrue ntrue-1 nfalse num_false-(nfalse-1) Pmiss(npts) Pfa(npts)]
    end

    #Pdet = Pdet(1:npts, 1:2);
    Pmiss = Pmiss(1:npts);
    Pfa   = Pfa(1:npts);

    return Pmiss, Pfa
    '''
