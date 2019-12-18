# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 06:42:10 2018
@author: arpita
"""

# The function performs the (non-linear) Kernel Fisher Analysis (KFA)
#
# PROTOTYPE
# model = perform_kfa_PhD(X, ids, kernel_type, kernel_args,n);
#
# USAGE EXAMPLE(S)
#
#     Example 1:
#       #In this example we generate the training data randomly and also
#       #generate the class labels manualy; for a detailed example have a
#       #look at the KFA demo
#       X=randn(300,100); #generate 100 samples with a dimension of 300
#       ids = repmat(1:25,1,4); #generate 25-class id vector
#       model = perform_kfa_PhD(X, ids, 'poly', [0 2], 24); #generate KFA model
#
#
# GENERAL DESCRIPTION
# The function computes the KFA subspace based on the training data X.
# Here, the data has to be arranged into the columns of the input training
# data matrix X. This means that if you have A inputs (input images in
# vector form) and each of these inputs has B elements (e.g., pixels), then
# the training-data matrix has to be of size BxA. The second input
# parameter "ind" represents a vector of class labels, where the class
# labels take numeric values. For example, if the first three images in X
# belong to the first subject and second three images belong to
# the second subject, and there would be 6 image in total in the matrix X,
# then the input parameter would take the following form: ind = [1 1 1 2 2
# 2]. In any case, the length of the parameter "ind" has to fit the nummber
# of smaples in X. The third and fourth input parameter define the kernel
# function used and the parameters of the kernel function. The fifth input
# parameter n determines the number of the retained KFA components. Here,
# the upper bounf on the number of components is given by the number of
# classes in X minus one.Note that n mustn't exceed the number of classes
# minus one, which equals the rank of the between class kernel matrix.
#
#
# The function does not perform any normalization of the input data, it just
# computes the KFA subspace. If you would like to perform data
# normalization (e.g., remove illumination variations) have
# a look at the demo folder and look up how that is done.
#
#
# REFERENCES
# The function is an implementation of the KFA technique described in:
#
# C. Liu: Capitalize on dimensionality increasing techniques for improving
# face recognition grand challenge performance, TPAMI, vol. 28, no. 5, pp.
# 725-737, May 2006.
#
#
#
# INPUTS:
# X                     - training-data matrix of size BxA, where each of
#                         the A columns contains one sample - each sample
#                         having a dimensionality of B (obligatory
#                         argument)
# ids                   - a vector of size 1xA, where each numeric value in
#                         "ids" enocedes the class-label of the corresponding
#                         sample in X (obligatory argument)
# kernel_type           - a string determining the type of the kernel;
#                         depending on the selected type appropriate kernel
#                         parameters have to used; valid strings for this
#                         argument are (optional argument):
#
#                         'poly' - the polynomial kernel, which requires
#                         two input arguments arranged in a 1x2 matrix in
#                         "kernel_args", (e.g., kernel_args = [0 2], which
#                         are also the defaults). If the second argument
#                         equals one, i.e., [0 1], then the polynomial
#                         kernel turns into the linear kernel, resulting in
#                         linear subspaces. The two arguments are used as
#                         follows:
#                           k(x,y)=(x'*y+kernel_args(1))^(kernel_args(2))
#
#                         'fpp' - the fractional power polynomial kernel,
#                         which requires two input arguments arranged in a
#                         1x2 matrix in "kernel_args", (e.g., kernel_args =
#                         [0 0.8], which are also the defaults).  The two
#                         arguments are used as follows:
#                           k(x,y)=sign(x'*y+kernel_args(1))*abs(x'*y+kernel_args(1))^(kernel_args(2))
#
#                         'tanh' - the sigmoidal kernel, which requires
#                         one numerical input argument in "kernel_args",
#                         (e.g., kernel_args = [0], which is also the
#                         default). The argument is used as follows:
#
#                               k(x,y)=tanh(x'*y+kernel_args(1))
# kerenl_args           - the parameters of the kernel function; their use
#                         is explained above (optional argument)
# n                     - the dimensionality of the KFA subspace (optional
#                         argument); if the parameter is not given, a default
#                         value is computed from the data in "ids"
#
# OUTPUTS:
# model                 - a structure containing the KFA subspace parameters
#                         needed to project a given test sample into the
#                         KFA subspace, the model has the following
#                         structure:
#
#                         model.K     . . . the training kernel matrix
#                         model.dim   . . . the dimensionality of the
#                                           KPCA subspace
#                         model.W     . . . the tranformation matrix (the eigenvectors)
#                         model.J     . . . the auxilary matrix needed for
#                                           centering the trainign data
#                         model.eigs  . . . the eigenvalues that can be
#                                           used for matching
#                         model.train . . . KPCA fetures corresponding to
#                                           the training data
#                         model.typ   . . . the string identifier of the
#                                           kernel type
#                         model.args  . . . the parameters of the kernel
#                                           function
#                         model.X     . . . the training data
#
#
#
# NOTES / COMMENTS
# The function is implemented to produce the KFA subspace. For an example
# of usage please have a look at the demo folder. Note also that this is
# not an implementation of the Generalized Discriminant Analysis, which is
# also a kernel extension of LDA. If you would like to use GDA, you have to
# look for another toolbox.
#
# The function was tested with Matlab ver. 7.9.0.529 (R2009b) and Matlab
# ver. 7.11.0.584 (R2010b).
#
#
#
# RELATED FUNCTIONS (SEE ALSO)
# perform_lda_PhD
# perform_kpca_PhD
# perform_pca_PhD
# linear_subspace_projection_PhD
# nonlinear_subspace_projection_PhD
#
#
# ABOUT
# Created:        15.2.2010
# Last Update:    29.11.2011
# Revision:       1.0
#
#
# WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
# OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
#
# Štruc V., Pavešic, N.: The Complete Gabor-Fisher Classifier for Robust
# Face Recognition, EURASIP Advances in Signal Processing, vol. 2010, 26
# pages, doi:10.1155/2010/847680, 2010.
#
# Štruc V., Pavešic, N.:Gabor-Based Kernel Partial-Least-Squares
# Discrimination Features for Face Recognition, Informatica (Vilnius), vol.
# 20, no. 1, pp. 115-138, 2009.
#
#
# The BibTex entries for the papers are here
#
# @Article{ACKNOWL1,
#     author = "Vitomir \v{S}truc and Nikola Pave\v{s}i\'{c}",
#     title  = "The Complete Gabor-Fisher Classifier for Robust Face Recognition",
#     journal = "EURASIP Advances in Signal Processing",
#     volume = "2010",
#     pages = "26",
#     year = "2010",
# }
#
# @Article{ACKNOWL2,
#     author = "Vitomir \v{S}truc and Nikola Pave\v{s}i\'{c}",
#     title  = "Gabor-Based Kernel Partial-Least-Squares Discrimination Features for Face Recognition",
#     journal = "Informatica (Vilnius)",
#     volume = "20",
#     number = "1",
#     pages = "115–138",
#     year = "2009",
# }
#
# Official website:
# If you have down-loaded the toolbox from any other location than the
# official website, plese check the following link to make sure that you
# have the most recent version:
#
# http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/PhDface/index.html
#
#
# OTHER TOOLBOXES
# If you are interested in face recognition you are invited to have a look
# at the INface toolbox as well. It contains implementations of several
# state-of-the-art photometric normalization techniques that can further
# improve the face recognition performance, especcially in difficult
# illumination conditions. The toolbox is available from:
#
# http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
#
#
# Copyright (c) 2011 Vitomir Štruc
# Faculty of Electrical Engineering,
# University of Ljubljana, Slovenia
# http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.
#
# November 2011


import numpy as np

from classification import KernelMatrix
from classification import Normo


## This is the auxilary function used to produce W
# for within-class scatter computation
def find(ele, arr):
	idx = []
	for i in range(0, len(arr)):
		if ele == arr[i]:
			idx.append(i)

	return idx


def return_W(W, ids):
	id_unique = np.unique(ids);
	# [c,d]=id_unique.shape
	num_of_class = len(id_unique)  # this is the number of classes

	for i in range(0, num_of_class):

		ind = find(id_unique[i], ids)

		[x, y] = np.meshgrid(ind, ind);

		elem_val = 1 / len(ind);
		l = len(x)
		for p in range(0, l):
			for q in range(0, l):
				W[x[p][q]][y[p][q]] = elem_val;

	return W


class Model(object):
	def __init__(self):
		dummy = 0


def perform_kfa_PhD(X, ids, kernel_type, n):
	## Init
	model = Model()  # model = [];

	## Check inputs

	# check number of inputs
	# if nargin <2
	#     disp('Wrong number of input parameters! The function requires at least two input arguments.')
	#     return;
	# elseif nargin >5
	#     disp('Wrong number of input parameters! The function takes no more than five input arguments.')
	#     return;
	# elseif nargin==2
	#     [a,b]=size(unique(ids));
	#     n = max([a,b])-1;
	#     kernel_type = 'poly'
	#     kernel_args = [0 2];
	# elseif nargin==3
	#     [a,b]=size(unique(ids));
	#     n = max([a,b])-1;
	if kernel_type == 'poly':
		kernel_args = [0, 2]
	elif kernel_type == 'fpp':
		kernel_args = [0, .8]
	elif kernel_type == 'tanh':
		kernel_args = 0;
	else:
		print('The entered kernel type was not recognized as a supported kernel type.')
		return

	# elseif nargin==4
	#     [a,b]=size(unique(ids));
	#     n = max([a,b])-1;
	# end
	#
	# #check if ids is a vector
	# if isvector(ids)==0
	#     disp('The second parameter "ids" needs to be a vector.')
	#     return;
	# end

	# check if n is not to big
	# [a,b]=size(unique(ids));
	# if n>max([a,b])-1;
	#     disp('The parameter "n" must not be larger than the number of classes minus one. Decreasing "n"!')
	#     n = max([a,b])-1;
	# end
	#
	# #check that each image in X has a class label
	[a, b] = X.shape
	'''
	may be not needed...
	if b~=length(ids)
		disp('The label vector "ids" needs to be the same size as the number of samples in X.')
		return;
	end
	'''

	# we assume that the data is contained in the columns - print to prompt
	# disp(sprintf('The training data comprises #i samples (images) with #i variables (pixels).', b, a))
	# disp('If this should be the other way around, please transpose the training-data matrix.')

	## Compute the KFA subspace - main part

	# compute the training data kernel matrix
	K = KernelMatrix.compute_kernel_matrix_PhD(X, X, kernel_type, kernel_args);

	model.K = K;

	# center kernel
	J = np.ones((b, b)) / b
	Kc = K - np.dot(J, K) - np.dot(K, J) + np.dot(np.dot(J, K), J)
	model.J = J

	# compute W using auxilary function
	W = np.zeros((b, b))

	W = return_W(W, ids)

	# construct eigenproblem using Tickhonov regularization
	epsi = 1e-10 * np.min(np.dot(Kc, Kc))  # some small regularization constant
	Crit = np.linalg.solve((np.dot(Kc, Kc) + epsi * np.identity(b)), (np.dot(np.dot(Kc, W), Kc)))
	del W

	# solve eigenproblem
	[U, V, L] = np.linalg.svd(Crit)
	del L
	del Crit

	Alpha = Normo.normc(U[:, 0:n])
	del U

	# normalize Alpha to be unit length in F

	R = np.dot(np.dot((Alpha.T), Kc), Alpha)
	# print(R)

	norms = np.real(np.diag(R))
	# print(norms)

	for i in range(0, n):
		Alpha[:, i] = Alpha[:, i] / (np.sqrt((norms[i])))

	# construct some outputs
	model.W = Alpha
	model.dim = n
	# print(V)
	# print(V.shape)
	# V=np.diag(V)
	# print(V)
	# print(V.shape)

	V = V.reshape(len(V), 1)

	model.eigs = V

	model.typ = kernel_type
	model.args = kernel_args
	model.X = X

	## Construct training features
	model.train = np.dot(Alpha.T, Kc)

	return model
