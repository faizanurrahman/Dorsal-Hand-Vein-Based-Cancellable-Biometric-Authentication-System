import csv
import os

import cv2
import numpy as np
import pandas as pd

from PreProcessing_FeatureExtraction.extract_feature import vein_pattern
from cancellability.feature_xortransform import transform_RG_XOR

# fix random seed for reproducibility
seed = 2
np.random.seed(seed)
# Load Random Matrix
Key = []

with open('Xor_Key.csv') as File:
	reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		for i in range(0, len(row)):
			row[i] = float(row[i])
		Key.append(row)

print('Key Matrix Loaded')
Key = np.array(Key)
# Key = Key.T
# Key = normalize(Key)


# Read File
input_folder_path = r'C:\Users\Faizan\Desktop\Dorsal Hand Vein Based Authentication System\DataSet\input\\'
input_folder_content = (os.listdir(input_folder_path))
print("Dataset contain", input_folder_content)
NumberOfSub = len(input_folder_content)
print("Number of input subfolder: ", NumberOfSub)
S_Path = []
for subfolderNo in range(0, NumberOfSub):
	S_Path.append(os.path.join(input_folder_path, input_folder_content[subfolderNo]))

sub_folder_content = []

for subfolderNo in range(0, NumberOfSub):
	sub_folder_content.append(os.listdir(S_Path[subfolderNo]))
# make directory for veinpattern
veinpattern_folder_path = "../sample dataset/veinpattern/"
veinpattern_path = []
for subfolder in range(NumberOfSub):
	new_path = os.path.join(veinpattern_folder_path, input_folder_content[subfolder])
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	veinpattern_path.append(new_path)

print("vein pattern directory created: ")

# make directory for xorfeature
xorfeature_folder_path = "../sample dataset/xorfeature/"
xorfeature_path = []
for subfolder in range(NumberOfSub):
	new_path = os.path.join(xorfeature_folder_path, input_folder_content[subfolder])
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	xorfeature_path.append(new_path)
print("Xorfeature directory created: ")

# make directory for transform_feature
transform_feature_folder_path = "../sample dataset/transform_feature/"
transform_feature_path = []
for subfolder in range(NumberOfSub):
	new_path = os.path.join(transform_feature_folder_path, input_folder_content[subfolder])
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	transform_feature_path.append(new_path)
print("transform_feature directory created: ")
# out_sub_folder_content = []

# for subfolderNo in range(0, NumberOfSubFolder):
#	out_sub_folder_content.append(os.listdir(O_Path[subfolderNo]))

# print(sub_folder_content[0][0])
R = len(sub_folder_content)  # No. of rows sub_folder_content[][]...40
C = len(sub_folder_content[0])  # No. of cols sub_folder_content[][]...10

# testing parameter for creating database
# R = 1
# C = 5
# stander size of image
k = 100
N = 100
# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
print("Dataset creation start....")
# DataMatrix
ID = []
for x in range(0, R):  # R: number of subjects
	print('Subject: ', x + 1)  # C : number of samples per subject
	key1 = Key[x].reshape(k, N)
	for y in range(0, C):
		ID.append(x)

		ImgPath = S_Path[x] + '\\' + sub_folder_content[x][y]
		# print('file:',ImgPath)
		Img = cv2.imread(ImgPath, 0)
		Img = np.asarray(Img, dtype=np.float64)
		# Img = Img[50:, 50:350]
		# Img = cv2.resize(Img, (k,N), interpolation=cv2.INTER_CUBIC)
		featurePattern = vein_pattern(Img, 6, 6)
		featurePattern = cv2.resize(featurePattern, (k, N), interpolation=cv2.INTER_CUBIC)
		print(f"{x, y} image processed: ")
		veinPattern = featurePattern
		cv2.imwrite(veinpattern_path[x] + '\\' + sub_folder_content[x][y], veinPattern)
		xorfeature = transform_RG_XOR(featurePattern, key1)
		cv2.imwrite(xorfeature_path[x] + '\\' + sub_folder_content[x][y], xorfeature)
		# transform_feature = feature_rdmtransform(veinPattern, key1, key2)
		# cv2.imwrite(transform_feature_path[x] + '\\' + sub_folder_content[x][y], transform_feature)
		# featurePattern = cv2.erode(featurePattern, element)
		# featurePattern = (featurePattern>0).astype(float)
		# featurePattern = morphology.thin(featurePattern)
		# transformedFeatureVector = XorAndMedian(featurePattern, RG)
		# cv2.imwrite(veinpattern_path[x] + '\\' + sub_folder_content[x][y], featurePattern)
		print(f"{x, y} image write: ")
		# temp=vein_pattern(Img, 5, 8)

		# transformedFeatureVector = transformedFeatureVector.ravel()
		transformedFeatureVector = xorfeature.ravel()
		#        print ('feature : ',fvs)

		# print((transformedFeatureVector[0]))
		if x == 0 and y == 0:
			dataMatrixRG = np.column_stack((transformedFeatureVector)).T
		else:
			dataMatrixRG = np.column_stack((dataMatrixRG, transformedFeatureVector))

print('Feature Extraction completed')

data = dataMatrixRG.T  # Transpose of dataMatrix
# print ('feature : ',data)
test_data = []
train_data = []
ids_test = []
ids_train = []

cnt = 0
for i in range(0, R):
	for j in range(0, C):
		if j == 3 or j == 2:
			test_data.append(data[cnt].astype(np.float64))
			ids_test.append(ID[cnt])
		else:
			train_data.append(data[cnt].astype(np.float64))
			ids_train.append(ID[cnt])
		cnt += 1

test_data = np.array(test_data)
train_data = np.array(train_data)
y_train = np.array(ids_train).T
y_test = np.array(ids_test).T

# test_data=((np.array(test_data)).T)
# train_data=((np.array(data)).T)
# creating csv file for all data
df_fullData = pd.DataFrame(data)
df_trainData = pd.DataFrame(train_data)
df_testData = pd.DataFrame(test_data)
df_yTrain = pd.DataFrame(y_train)
df_yTest = pd.DataFrame(y_test)

df_fullData.to_csv('full_data.csv', index=False, header=False)
df_trainData.to_csv('Xtrain.csv', index=False, header=False)
df_testData.to_csv('Xtest.csv', index=False, header=False)
df_yTrain.to_csv('y_train.csv', index=False, header=False)
df_yTest.to_csv('y_test.csv', index=False, header=False)

print('database created succesfully.')
