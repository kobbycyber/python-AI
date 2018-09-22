import numpy as np      # np is an alias pointing to numpy
from sklearn import preprocessing
input_data = np.array(
                    [[2.1, -1.9, 5.5], 
		            [-1.5, 2.4, 3.5], 
                    [0.5, -7.9, 5.6], 
                    [5.9, 2.3, -5.8]])

print("Input data:\n", input_data)
print("\n")

data_binarized = preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("Binarized data:\n", data_binarized)
print("\n")

print("input_data.Mean = ", input_data.mean(axis = 0))
print("input_data.Std deviation = ", input_data.std(axis = 0))
print("\n")

data_scaled = preprocessing.scale(input_data)
print("data scaled:\n", data_scaled)
print("\n")
print("data_scaled.Mean = ", data_scaled.mean(axis=0))
print("data_scaled.Std deviation = ", data_scaled.std(axis = 0))
print("\n")

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print ("Min max scaled data:\n", data_scaled_minmax)
print("\n")

# Normalize data 
data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1') 
print("\nL1 normalized data:\n", data_normalized_l1)
print("\n")

# Normalize data
data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')
print("\nL2 normalized data:\n", data_normalized_l2)