import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('dataCleaned.csv')
#xóa các thuộc tính không sử dụng vào việc dào tạo mô hình
#,'Behavior','Treatment','Genotype'
dataNew = data.drop(['MouseID'],axis=1)
dataset = dataNew.drop(['class',"Treatment","Behavior", "Genotype"],axis=1)
#dataset = pd.get_dummies(data=dataset,columns=["Treatment","Behavior", "Genotype"])

#dataset = dataset.replace({'Control':0,'Ts65Dn':1,'Memantine':0,'Saline':1,'C/S':0,'S/C':1})
target = dataNew['class']
target = target.replace({'c-SC-m': 0, 'c-CS-m':1, 't-SC-m':2, 't-CS-m':3, 't-SC-s':4, 'c-SC-s':5, 'c-CS-s':6, 't-CS-s':7})

x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                   target,
                                                   test_size=0.3,
                                                   stratify=target.values,
                                                  random_state=999)

print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# KNN
neighbors = np.arange(1,10)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #đào tạo model
    knn.fit(x_train, y_train)
    
    #độ chính xác tập train
    train_accuracy[i] = knn.score(x_train, y_train)
    
    #độ chính xác tập test
    test_accuracy[i] = knn.score(x_test, y_test)

    
#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

## chon k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
precision_knn  = metrics.accuracy_score(y_pred, y_test)*100
f1_knn = f1_score(y_test, y_pred, average='macro')
print('Accuracy with K-NN: {0:.2f}% '.format(precision_knn))
print(confusion_matrix(y_test,y_pred))
test = x_test.iloc[100]
rs_test = knn.predict([test])
print(rs_test)
#print('f1_score:  ', f1_knn)

# SVM
svm = SVC(gamma='auto', kernel='linear')
svm.fit(x_train, y_train) 
y_pred_svm = svm.predict(x_test)
precision_svm = metrics.accuracy_score(y_test,y_pred_svm) * 100
print("Accuracy with SVM: {0:.2f}%".format(precision_svm))
print(confusion_matrix(y_test,y_pred_svm))
# randomforest
from sklearn.ensemble import RandomForestClassifier
randomfr = RandomForestClassifier(n_estimators=200,max_depth=8, random_state=33)
randomfr.fit(x_train, y_train)
pred_y_fr = randomfr.predict(x_test) 
precision_fr  = metrics.accuracy_score(pred_y_fr, y_test)*100
print("Accuracy with RandomForest: {0:.2f}%".format(precision_fr))
print(confusion_matrix(y_test,pred_y_fr))

'''
#logistic

# Making an instance of the model
lr = LogisticRegression()

# fitting the model to the training data
lr.fit(x_train, y_train)

# use the model to predict on the testing data
pred_y_lr = lr.predict(x_test)

# Printing the accuracy of the model
score = metrics.accuracy_score(pred_y_lr, y_test)*100
'''

import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# Making a dataframe of the accuracies
a = { 'K-Nearest Neighbours': [precision_knn], 'Random Forest Classifier': [precision_fr]}
accuracies = pd.DataFrame(data=a)
#accuracies.rename(index={0:'Random Forest Classifier',1:'K-Nearest Neighbours', 2:'Logistic Regression'}, 
#                 inplace=True)

# making bar plot comparing the accuracies of the models
sns.set(font_scale=1)
ax = accuracies.plot.bar(
    figsize= (13, 5),
    fontsize=14)
plt.xticks(rotation=0, fontsize=14)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
x_labels = ['A', 'B']
xticks = [-0.17,0.165]
ax.set_xticks(xticks)
ax.set_xticklabels(x_labels, rotation=0)
axbox = ax.get_position()
plt.legend(loc = (axbox.x0 + 0.65, axbox.y0 + 0.70), fontsize=14)
plt.title(' ')
ax.set_facecolor('xkcd:white')
ax.set_facecolor(('#ffffff'))
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.show()
