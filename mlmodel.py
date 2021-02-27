
'''
KNN model for fraud detection system
'''

import pandas as pd
import sys
import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv("creditcard.csv")

# take a look at the dataset
#df.head()
# Import train_test_split function
from sklearn.model_selection import train_test_split
#use required features
cdf = df[['Time','V1','V2','V3','Amount','Class']]
#,'V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28'

#print(cdf.loc[cdf['Class'] == 1])
#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :5]
y = cdf.iloc[:, -1]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
#regressor = LinearRegression()
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)
#Fitting model with trainig data
#regressor.fit(x, y)
knn.fit(X_train,y_train)
# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
#pickle.dump(regressor, open('model.pkl','wb'))
pickle.dump(knn, open('model.pkl','wb'))
'''
#Predict the response for test dataset
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
'''
'''
y_pred = knn.predict(x)
print(y_pred)
'''
'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(x))
#print(model.predict([[7519,	1.234235046,	3.019740421,	-4.304596885,	4.73279513,	3.624200831,	-1.357745663,	1.713444988,	-0.496358487,	-1.28285782,	-2.447469255,	2.101343865,	-4.609628391,	1.464377625,	-6.079337193,	-0.339237373,	2.581850954,	6.739384385,	3.042493178,	-2.721853122,	0.009060836,	-0.379068307,	-0.704181032,	-0.656804756,	-1.632652957,	1.488901448,	0.566797273,	-0.010016223,	0.146792735,	1
#]]))
'''



