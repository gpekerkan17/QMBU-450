import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# In this project I will try to find the best model to predict wine quality

#let's get started by reading the data
wine = pd.read_csv("winequality.csv")
#let us look how the data is distributed
wine.head(7)

wine.columns #to get a clear perspective of all the metrics

wine.info()

wine.describe()

wine['quality'].value_counts()  #shows number of different quality types


"""Creating a heatmap to show correlation"""

corr = wine.corr()
fig = plt.subplots(figsize=(15,10))
sns.heatmap(corr,square=True,annot=True,cmap='Reds') # Here is a nice heatmap


"""Correlation Matrix"""
#if heatmap does not show a good perspective look at the heatmap
rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = wine.corr()
corr.style.background_gradient(cmap='coolwarm')


"""Classifying 'quality' into binaries"""

reviews = []
for i in wine['quality']:
    if i <= 6:
        reviews.append('0') #0 is considered bad quality
    elif i >= 7 :
        reviews.append('1') #1 is considered good quality
wine['quality'] = reviews

sns.pairplot(wine, hue="quality", palette="husl")

wine['quality'].value_counts() #counts number of 0s and 1s


#Separating the dataset: splitting X and y variables
X = wine.drop('quality', axis=1)
y = wine['quality']
#to be sure we can look at the shapes
X.shape
y.shape

"""Train, Test, Split - Model """
#classification
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)
Xtrain.shape
Xtest.shape
ytrain.shape
ytest.shape

"""Gaussian NB model"""

model = GaussianNB()
#fit the data
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

"Accuracy score: ",accuracy_score(ytest, y_model)*100



"""Decision Tree classification"""

dtc = DecisionTreeClassifier()

dtc = dtc.fit(Xtrain,ytrain)
ypred = dtc.predict(Xtest)

"Accuracy:",metrics.accuracy_score(ytest, ypred)*100



"""Support Vector Machines """

svc = SVC()

svc.fit(Xtrain,ytrain)
pred_svc =svc.predict(Xtest)

classification_report(ytest,pred_svc)



"""Random Forest Classifier"""

rfc = RandomForestClassifier(n_estimators=250)

rfc.fit(Xtrain, ytrain)

pred_rfc = rfc.predict(Xtest)
classification_report(ytest, pred_rfc)



""" KNeighbors Classifier """

knn = KNeighborsClassifier()

knn.fit(Xtrain,ytrain)
pred_knn=knn.predict(Xtest)
classification_report(ytest, pred_knn)
