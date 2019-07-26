# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:54:25 2019

@author: Harshit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd
import seaborn as sns



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
count=train['domain'].value_counts()
count2=train['domain'].value_counts().count()
count.index
count.values
train['domain'].astype('category').cat.categories.tolist()
sns.set(style="darkgrid")
sns.barplot(count.index,count.values,alpha=0.9)
plt.show()

train['title'].fillna('', inplace=True)
test['title'].fillna('', inplace=True)


dataset = train.append(test)
dataset.info()

X=dataset.iloc[:,6:].values
X=pd.DataFrame(X)
(X).apply(lambda x: sum(x.isnull()),axis=0)
X[0].fillna('', inplace=True)

train.info()

train.apply(lambda x: sum(x.isnull()),axis=0)
sum(train['title'].isnull())


#if we need to drop nan values of some particular columns
train.dropna(subset=['title'],inplace=True)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,66876):    
    review=re.sub('[^a-zA-Z]',' ',X[0][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #using set to increse the speed of the algorithm
    review=' '.join(review)
    corpus.append(review)

      
#creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
corpusarray=cv.fit_transform(corpus).toarray()
data=pd.DataFrame(corpusarray)
Train = data.iloc[0:50000,:].values
Test = data.iloc[50000:66876,:].values
Train=pd.DataFrame(Train)  
Test=pd.DataFrame(Test) 
Train['over_18']=train['over_18']
Test['over_18']=test['over_18']


y=train.iloc[:,6:]

df=pd.DataFrame(X)
df['over_18']=train['over_18']
df['domain']=train['domain']
df2=pd.DataFrame(X_test2)
df2['over_18']=test['over_18']
df2['domain']=test['domain']

del df['domain']
del df2['domain']    

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Train, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca=PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 10, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)#no variance calculation

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Fitting classifier2 to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion ='entropy',random_state=0)
classifier.fit(X_train,y_train)


# Fitting classifier3 to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,random_state=0,criterion='entropy')
classifier.fit(X_train,y_train)

# Fitting classifier3 to the full training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,random_state=0,criterion='entropy')
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_predtestNB=classifier.predict(Test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm