# step 1: import Scikit-learn
import sklearn
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# step 2: import scikit-learn dataset. 
# This load_breast_cancer is the database of breast cancer from Wisconsin
from sklearn.datasets import load_breast_cancer

# load the dataset
data = load_breast_cancer()

# Create variable for each dataset
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# print the class label
print('label_names: ' , label_names)
# print label number
print('labels[0]: ' , labels[0])
# print feature name and values
print('feature_names[0]: ' , feature_names[0])
print('features[0]: ')
print(features[0])

# step 3: orginize data into set: import split function
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = \
train_test_split(features,labels,test_size = 0.40, random_state = 42)

# step 4: build the model: Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

# Step 5: evaluate the model and its accuracy
preds = gnb.predict(test)
print ('preds: ')
print(preds)
from sklearn.metrics import accuracy_score
print ('accuracy_score: ', accuracy_score(test_labels,preds))