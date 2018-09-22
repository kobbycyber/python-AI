import numpy as np      # np is an alias pointing to numpy
from sklearn import preprocessing
import warnings
# Sample input labels
input_labels = ['red','black','red','green','black','yellow','white']
print("\ninput_labels =", input_labels)

# Creating the label encoder 
encoder = preprocessing.LabelEncoder() 
encoder.fit(input_labels)

# encoding a set of labels
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
print ("\n")

# decoding a set of values reverse back to labels
encoded_values_1 = [3,0,4,2,1]   # white, blacm yellow, red, green
print("Encoded values 1 =", encoded_values_1)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
decoded_list_1 = encoder.inverse_transform(encoded_values_1)
print("Decoded labels 1 =", list(decoded_list_1))