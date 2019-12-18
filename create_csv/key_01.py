import csv
import random

# Create of Key Matrix [10000x85]

K = 100
N = 100
R = 85  # Number of persons
C = 10  # Number of sample
Key = []
for i in range(0, R):
	Key.append([])
	for j in range(0, K * N):
		key_cell_value = (random.randint(10, 50))
		Key[i].append(key_cell_value)

print('Key created')

myFile = open('Key01.csv', 'w', newline='')
with myFile:
	writer = csv.writer(myFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	writer.writerows(Key)

print("Writing complete")
