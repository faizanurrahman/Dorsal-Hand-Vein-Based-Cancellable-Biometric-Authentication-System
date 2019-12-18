print('\n')
print('Load images from a database and compute Gabor magnitude features.')
print('During this step we also need to  construct the \n Gabor filter bank and extract the Gabor magnitude features.')
print('This may take a while.')

import csv
import os
# from Classification import Results
import warnings

import cv2
import numpy as np
import transformUsingMeximumCurvatureAndRG as MCRG
from Classification import KFA
from Classification import Projection
from scipy.spatial import distance
# from Classification import Results
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


def minmax(X, low, high, minX=None, maxX=None, dtype=np.float):
	X = np.asarray(X)
	if minX is None:
		minX = np.min(X)
	if maxX is None:
		maxX = np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	# X = X * (high-low)
	# X = X + low
	return np.asarray(X, dtype=dtype)


main_folder_path = r"C:\Users\Faizan\Desktop\PBI_Dorsal_vein_Biometric_Authentication\Database\DorsalVeins Image"
main_folder_content = (os.listdir(main_folder_path))
NumberOfSub = len(main_folder_content)
S_Path = []
for subfolderNo in range(0, NumberOfSub):
	S_Path.append(main_folder_path + '\\' + main_folder_content[subfolderNo])
sub_folder_content = []
for subfolderNo in range(0, NumberOfSub):
	sub_folder_content.append(os.listdir(S_Path[subfolderNo]))

R = len(sub_folder_content)  # No. of rows sub_folder_content[][]...85
C = len(sub_folder_content[0])  # No. of cols sub_folder_content[][]...10
R = 5
C = 2
k = 100
N = 100
KeyMat = []

with open(
		r'C:\Users\Faizan\Desktop\Intern-Arpita-HK_Codes\Intern-Arpita-HK_Codes\Scheme 1- Random Grid ,LG, XOR,Median\Key09.csv') as File:
	reader = csv.reader(File, delimiter=',', quotechar=',',
	                    quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		for i in range(0, len(row)):
			row[i] = float(row[i])
		KeyMat.append(row)

print('Key Matrix Loaded')
KeyMat = np.array(KeyMat)
KeyMat = KeyMat.T
KeyMat = normalize(KeyMat)
KeyMat1 = []

with open(
		r'C:\Users\Faizan\Desktop\Intern-Arpita-HK_Codes\Intern-Arpita-HK_Codes\Scheme 1- Random Grid ,LG, XOR,Median\Key10.csv') as File:
	reader = csv.reader(File, delimiter=',', quotechar=',',
	                    quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		for i in range(0, len(row)):
			row[i] = float(row[i])
		KeyMat1.append(row)

print('Key Matrix 2 Loaded')
KeyMat1 = np.array(KeyMat1)
KeyMat1 = KeyMat1.T
KeyMat1 = normalize(KeyMat1)

fvs3 = []
# DataMatrix
ID = []
for x in range(0, R):  # R: number of subjects
	print('Subject: ', x + 1)  # C : number of samples per subject
	for y in range(0, C):
		ID.append(x)

		ImgPath = S_Path[x] + '\\' + sub_folder_content[x][y]
		###print('file:',ImgPath)
		Img = cv2.imread(ImgPath, 0)

		####print('===========Img=======================')
		###print (Img)
		Img = np.asarray(Img, dtype=np.float64)
		# Img = cv2.resize(Img, (k,N), interpolation=cv2.INTER_CUBIC)

		####print('===========key=======================')
		Key = KeyMat[x]
		###print (Key )
		# For worst case scenario (same key for all users) set x=1 or x=2 as required...
		####print('===========Key1=======================')
		Key1 = KeyMat1[x]
		print(Key1)
		Key = Key.reshape(k * N, 1)
		# Key=Key.reshape(k*N*nscale*norient,1)
		Key1 = Key1.reshape(k * N, 1)
		# transformedFeatureVector=Img
		transformedFeatureVector, fvs = MCRG.transformUsingMeximumCurvatureAndRG(Img, Key, Key1, 4, k,
		                                                                         N)  # (24x100x100,1)
		#####print('===========fvs=======================')
		#####print ('feature : ',fvs)

		# print((transformedFeatureVector[0]))
		if x == 0 and y == 0:
			dataMatrixRG = np.column_stack((transformedFeatureVector)).T
		else:
			dataMatrixRG = np.column_stack((dataMatrixRG, transformedFeatureVector))

print('Feature Extraction completed')
print('===========fv train=======================')
data = dataMatrixRG.T  # Transpose of dataMatrix
print('feature : ', data)
print('==================================')

ids_train = []
# ids_test=5
cnt = 0
for i in range(0, R):
	for j in range(0, C):
		ids_train.append(ID[cnt])
		cnt += 1

# test_data=((np.array(test_data)).T)
train_data = ((np.array(data)).T)

model = KFA.perform_kfa_PhD(train_data, ids_train, 'fpp', len(ids_train))
fvs2 = []
k = 100
N = 100
R1 = 1
C1 = 1
ids_test = 0

for x in range(0, R1):
	for y in range(0, C1):

		ImgPath = S_Path[x] + '\\' + sub_folder_content[x][5]
		#        Img2=cv2.imread('E:/Dhruv/Database/Dorsal-veins/s05/2017213_R_0.jpg',0)
		#  Img2=cv2.imread(r'C:\Users\Faizan\Desktop\PBI_Dorsal_vein_Biometric_Authentication\Database\DorsalVeins Image\s01\2017232_R_0.jpg'
		Img2 = cv2.imread(ImgPath, 0)

		###print('===========Img2=======================')
		###print (Img2)
		###print('===============Key===================')
		Img2 = np.asarray(Img2, dtype=float)
		# Img2 = cv2.resize(Img2, (k,N), interpolation=cv2.INTER_CUBIC)

		Key = KeyMat[ids_test]
		###print (Key)
		###print('===========Key1=======================')
		# For worst case scenario (same key for all users) set x=1 or x=2 as required...
		Key1 = KeyMat1[ids_test]
		###print (Key1)
		Key = Key.reshape(k * N, 1)
		Key1 = Key1.reshape(k * N, 1)
		# transformedFeatureVector=Img
		transformedFeatureVector1, fvs1 = MCRG.transformUsingMeximumCurvatureAndRG(Img2, Key, Key1, 4, k,
		                                                                           N)  # (24x100x100,1)
		###print('===========fvs1=======================')
		###print ('feature : ',fvs1)

		# log=lgc.logGaborConvolve(Img2, filterbank_freq,nscale,norient)
		# print((transformedFeatureVector[0]))
		if x == 0 and y == 0:
			dataMatrixRG1 = np.column_stack((transformedFeatureVector1)).T
		else:
			dataMatrixRG1 = np.column_stack((dataMatrixRG1, transformedFeatureVector1))

# print(len(transformedFeatureVector))
# print(featureVector[800][0])

# print(len(dataMatrixRG))
# print(len(dataMatrixRG[200]))
print('Feature Extraction completed')
print('===========fv test=======================')
data1 = np.array(dataMatrixRG1).T  # Transpose of dataMatrix
print('feature : ', data1)
test_data = ((np.array(data1)).T)
testfeature = Projection.nonlinear_subspace_projection_PhD(test_data, model)
dt = model.train
print(testfeature)
for x in range(0, len(ids_train)):
	u = np.array(dt[:, x])
	v = np.array(testfeature)

	di = distance.euclidean(v, u)
	#    di = scipy.spatial.distance.pdist((u,v), 'euclidean')
	#    [di] = np.sqrt((v-u)**2)
	#    di=di.mean()
	print('value of dist=  ', di)
	# dist=pdist(p, 'euclidean')
	#
	if di <= 100:
		# print'access granted for: ',x
		print('acess granted for id no : ', ids_train[x] + 1)
	else:
		print('sorry: ', ids_train[x] + 1)