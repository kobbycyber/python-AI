# SVM (Support Vector Machines) for regression and classification
# pip3 install pandas
import pandas as pd
import numpy as np
from sklearn import svm, datasets
# pip3 install matplotlib
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# Get iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
#print ('X: ', X)
y = iris.target
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
C = 1.0
svc_classifier = svm.SVC(kernel='linear', C=C, decision_function_shape = 'ovr').fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize = (15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)
plt.xlabel('Sepal length') # sepal is a piece of flower
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max()) 
plt.show()