import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,SGDClassifier
from itertools import cycle
from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

## Step 1: Load data from a csv file
data = pd.read_csv("online_shoppers_intention.csv",header=0)

## Step 2: Data explosion
data.shape # shape of the data(number of rows vs number of column)
data.info # Information of data
data.describe() # description of the data
data.isnull().sum() #No missing value found
N_Data = data

VisitorType={'Returning_Visitor':1, 'New_Visitor':2, 'Other':3} #Replace text as numbers
N_Data['VisitorType']=N_Data['VisitorType'].map(VisitorType) #map into dataframe

Month={'Feb':2, 'Mar':3, 'May':5, 'Oct':10, 'June':6, 'Jul':7, 'Aug':8, 'Nov':11, 'Sep':9,'Dec':12} #replace month as numbers
N_Data['Month']=N_Data['Month'].map(Month) #map into dataframe

TF={True:1, False:0}
N_Data['Weekend']=N_Data['Weekend'].map(TF)
N_Data['Revenue']=N_Data['Revenue'].map(TF)

Corr = N_Data.corr()
fig, ax =plt.subplots(figsize=(15,15))  
sns.heatmap(Corr, xticklabels=Corr.columns, yticklabels=Corr.columns, annot=True)

# Correlation with Revenue
Revenue_corr = data.corr()['Revenue'] 
fig, ax=plt.subplots(figsize=(8,5)) 
sns.barplot(Revenue_corr[0:-1].index,Revenue_corr[0:-1].values,palette="Blues_d").set_title('Correlation with Revenue',fontsize = 25)
plt.xticks(rotation = 90)
plt.show()

#From the heat map, we can see that there isn't much correlation between features. Bounce rate and Exiterate has a corr as 0.91. 
#Productrelated and prodcuctrelated_duration has a corr as 0.86.  Informational and informational_duration has a corr as 0.62. Administraive and administrative_duration has a corr as 0.6. 
#However, these three is colsely related since people who spent more some on these categories definitely viewed more pages of these categories.

#Visual Exploratory Data Analysis 

# Administrative_Duration vs Revenue
plt.rcParams['figure.figsize'] = (8, 5)
sns.boxenplot(data['Administrative_Duration'], data['Revenue'], palette = 'pastel', orient='h')
plt.title('Admin. duration vs Revenue', fontsize = 30)
plt.xlabel('Admin. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
# From the plot we can find that the distributions of Administrative duration are bell-shaped for both purchased of not purchased.
# But there more outliers in the not purchased distribution.

# Informational_Duration vs Revenue
plt.rcParams['figure.figsize'] = (8, 5)
sns.boxenplot(data['Informational_Duration'], data['Revenue'], palette = 'pastel', orient='h')
plt.title('Informational. duration vs Revenue', fontsize = 30)
plt.xlabel('Informational. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
# From the plot we can find that the distributions of Information duration are bell-shaped for both purchased of not purchased.
# But there more outliers in the not purchased distribution.

# ProductRelated_Duration vs Revenue
plt.rcParams['figure.figsize'] = (8, 5)
sns.boxenplot(data['ProductRelated_Duration'], data['Revenue'], palette = 'pastel', orient='h')
plt.title('ProductRelated. duration vs Revenue', fontsize = 30)
plt.xlabel('ProductRelated. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
# From the plot we can find that the distributions of ProductRelated_Duration are bell-shaped for both purchased of not purchased.
# But there more outliers in the not purchased distribution.

# BounceRates VS ExitRates
X1 = data['BounceRates']
X2 = data['ExitRates']
X = pd.concat([X1,X2],axis=1).values
Y = data['Revenue']
label_0 = []
label_1 = []
for i in range(Y.shape[0]):
    if(Y[i]==0):
        label_0.append(X[i])
    else:
        label_1.append(X[i])
res_0 = np.array(label_0)
res_1 = np.array(label_1)
plt.scatter(res_0[:,0],res_0[:,1])
plt.scatter(res_1[:,0],res_1[:,1])
plt.xlabel('BounceRates', fontsize = 15)
plt.ylabel('ExitRates', fontsize = 15)
plt.show()
# From the plot, we find that people with low BounceRate and low ExitRate tend to purchase.

# Page value
plt.rcParams['figure.figsize'] = (8, 5)
sns.stripplot(data['PageValues'], data['Revenue'], palette = 'spring', orient = 'h')
plt.title('Page Values vs Revenue', fontsize = 30)
plt.xlabel('PageValues', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
# From the plot, we find there are many outliers for both purchased-true part and purchased-false part.
# However, we find that the page value affect the purchased-true part highly.

# Special Day
data['SpecialDay'].value_counts()
size = [11079,351,325,243,178,154]
colors = ['lightblue', 'green', 'pink','orange','darkblue','red']
labels = '0.0', '0.6', '0.8','0.4','0.2','1.0'
explode = (0,0,0,0,0,0)  

plt.subplots()
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.3f%%', shadow=False)
plt.title("Special Day",fontsize = 25)
plt.axis('off')
plt.legend()
plt.show()
# We can see from the pie chart that the data distribution of SpecialDay is highly imbalanced. 89.85% of the user was browsing on the days that were ot close to any special days.

# Month
data['Month'].value_counts()
size = [3364,2998,1907,1727,549,448,433,432,288,184]  
colors = ['lightGreen', 'green', 'pink','orange','lightblue','red','darkred','blue','violet','yellow']
labels = "May","November","March","December","October","September","August","July","June","February"
explode = [0,0,0,0,0,0,0,0,0,0]

circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.pie(size, colors = colors, explode = explode,labels = labels, shadow = False, autopct = '%.1f%%')
plt.title('Month', fontsize = 25)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show()

#We can see that March, May, November, December are the months of the year that users view the page more often.

# Operating Systems
data['OperatingSystems'].value_counts()
size = [6601,2585,2555,478,111]
colors = ['yellow', 'orange', 'pink', 'lightblue', 'lightgreen']
labels = "2","1","3","4","others"
explode = (0,0,0,0,0)  

plt.subplots()
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.3f%%', shadow=False)
plt.title("Operating Systems",fontsize = 25)
plt.axis('off')
plt.legend()
plt.show()
#We can see that 99% of the users are using the top four operating systems and the top operating system has a high share of 53.5%.

# Browser
data['Browser'].value_counts()
size = [7961,2462,736,467,174,163,135,232]
colors = ['orange', 'yellow', 'pink', 'red', 'lightgreen', 'green', 'cyan','lightblue']
labels = "2","1","4","5","6","10","8","others"
explode = (0,0,0,0,0,0,0,0)

plt.subplots()
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.3f%%', shadow=False)
plt.title("Browser",fontsize = 25)
plt.axis('off')
plt.legend()
plt.show()
#We can see that the data distribution is imbalanced and about 85% of the users are using the top two browser.

# Region
data['Region'].value_counts()
size = [4780,2403,112,1136,805,761,511,434,318]
colors = ['orange', 'yellow', 'pink', 'red', 'lightgreen', 'green', 'cyan','lightblue','darkgreen']
labels = "1","3","4","2","6","7","9","8","5"
explode = (0,0,0,0,0,0,0,0,0)

circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.pie(size, colors = colors, explode = explode,labels = labels, shadow = False, autopct = '%.3f%%')
plt.title('Region', fontsize = 25)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show()
#We can see that the company can focus more on users from region 1, 3 and 2 since 70% of the users come from these three regions.

# Traffic Type
data['TrafficType'].value_counts()
size = [3913,2451,2052,1069,738,450,444,1213]
colors = ['lightblue', 'lightgreen', 'pink', 'red', 'orange', 'green', 'cyan','darkgreen']
labels = "2","1","3","4","13","10","6","others"
explode = (0,0,0,0,0,0,0,0)

plt.subplots()
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.3f%%', shadow=False)
plt.title("Traffic Type",fontsize = 25)
plt.axis('off')
plt.legend()
plt.show()
#We can see that the company can focus more on users who use traffic type 2, 1, and 3 since 69% of the users come from these three traffics.

# Visitor Type
data['VisitorType'].value_counts()
size = [10551, 1694, 85]
colors = ['lightblue', 'lightgreen', 'pink']
labels = "Returning Visitor", "New Visitor", "Others"
explode = [0, 0, 0]

plt.subplots()
plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%.3f%%', shadow=False)
plt.title("Visitor Type",fontsize = 25)
plt.axis('off')
plt.legend()
plt.show()

#We can see that the data distribution is highly imbalanced. Returning Visitor has 85% of the total visitors. The company should put more focus on returning visitors to increase revenue.

# Weekend
data['Weekend'].value_counts()
fig = plt.figure(figsize =(10, 7))
labels = ['0','1']
size = [9462,2868]
plt.bar(labels, size, color ='lightgreen', width = 0.4)
plt.xlabel("Weekend or Weekday")
plt.ylabel("Number of users visited")
plt.title("Weekend")
plt.show()
#We can see that the data distribution is highly imbalanced. More users shop on weekdays than on weekends.

# Revenue
data['Revenue'].value_counts()
fig = plt.figure(figsize =(10, 7))
labels = ['0','1']
size = [10422,1908]
plt.bar(labels, size, color ='green', width = 0.4)
plt.xlabel("Buy or not")
plt.ylabel("Number of users")
plt.title("Revenue")
plt.show()
#We can see that the data distribution is highly imbalanced. Numbers of users who decided to not buy is five times greater than the people who decided to buy.

data.hist(bins=40, figsize=(20,15))
plt.show()
# From the histogram we can see that the data values' distribution is highly imbalance.

## Step 3: data transformation
data = pd.read_csv("online_shoppers_intention.csv",header=0)
dict_vec = DictVectorizer(sparse=False)
mydata = dict_vec.fit_transform(data.to_dict(orient = 'record'))
print(dict_vec.feature_names_)

## Step 4: data normalization
def rescaleMatrix(dataMatrix):
    colCount = len(dataMatrix[0])
    rowCount = len(dataMatrix)
    newMatrix = np.zeros(dataMatrix.shape) 
    for i in range(0, colCount):
        min = dataMatrix[:,i].min()
        denom = dataMatrix[:,i].max() - min
        for k in range(0, rowCount):
            newX = (dataMatrix[k,i] - min) / denom
            newMatrix[k,i] = newX
    return newMatrix
Y_label = mydata[:,22:23]
X_features = np.delete(mydata,22, axis = 1)
newdata = rescaleMatrix(X_features)


## Step 5: split the data and define functions
n = 12330
S = np.random.permutation(n)

# 9000 training samples
X_train = X_features[S[:9000],]
Y_train = Y_label[S[:9000],]

# 3330 testing samples
X_test = X_features[S[9000:],]
Y_test = Y_label[S[9000:],]

# define functions
# ROC curve
def roc(y_score):
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    fpr[1], tpr[1], _ = roc_curve(Y_test, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure(figsize=[9,7])
    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)
    plt.plot([1,0], [1,0], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('false positive rate', fontsize=18)
    plt.ylabel('true positive rate', fontsize=18)
    plt.title('ROC curve for purchase', fontsize=18)
    plt.legend(loc='lower right')
    plt.show()

# Confusion Matrix
def plot_confusion(prediction):
    conmat = np.array(confusion_matrix(Y_test, prediction, labels=[1,0]))
    confusion = pd.DataFrame(conmat, index=['purchase_ture', 'purchase_false'], 
                             columns=['predicted purchase_ture', 'predicted purchase_false'])
    print (confusion)

# Sigmoid Function
def Sigmoid(x):
	g = 1/(1+np.exp(-x))
	return g

## Step 6 : train models
# model1: logistic regression(basic linear model)
model1 = LogisticRegression()
model1.fit(X_train,Y_train)
print(model1.score(X_test,Y_test))
y_score1= model1.decision_function(X_test)
# roc curve
roc(y_score1)
# confusing matrix
prediction1 = model1.predict(X_test)
plot_confusion(prediction1)
# Show the recall and precision
print (classification_report(Y_test, prediction1, target_names=['purchase_false', 'purchase_true']))


# model2: logistic regression(regularized with Ridge)
sgd = SGDClassifier(loss='log', penalty='l2', learning_rate='optimal')
sgd_params = {'alpha': [1], 'class_weight': [ 'balanced']}
rfecv = RFECV(estimator=sgd, scoring='roc_auc')
model2 = rfecv.fit(X_train, Y_train)
prediction2 = model2.predict(X_test)
y_score2 = model2.decision_function(X_test)
print ('Test score: ', model2.score(X_test, Y_test))
# Plot ROC curve
roc(y_score2)
# Print confusion matrix
plot_confusion(prediction2)
# Show the recall and precision
print (classification_report(Y_test, prediction2, target_names=['purchase_false', 'purchase_true']))


# model3: neural network
model3 = keras.Sequential()
model3.add(layers.Dense(150, activation='relu', input_shape=(28,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(150, activation='relu', input_shape=(150,)))
model3.add(layers.Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.001)
model3.compile(loss=keras.losses.MeanSquaredError(),optimizer = opt)
model3.fit(X_train,Y_train,epochs=100)
Y_predict3 = model3.predict(X_test)
Y_predict3 = Y_predict3.flatten()
y_score3= model3.predict(X_test)
# roc curve
roc(y_score3)
# confusing matrix
hatProb = Sigmoid(Y_predict3)
prediction3 = (hatProb >= 0.65).astype(int) 
plot_confusion(prediction3)
# Show the recall and precision
print (classification_report(Y_test, prediction3, target_names=['purchase_false', 'purchase_true']))


# model4: SVM
# choose the best C 
model4 = SVC(kernel="rbf")
model4 = model4.fit(X_train, Y_train)
prediction4 = model4.predict(X_test)
y_score4 = model4.decision_function(X_test)
print ('Test score: ', model4.score(X_test, Y_test))
# roc curve
roc(y_score4)
# confusing matrix
plot_confusion(prediction4)
# Show the recall and precision
print (classification_report(Y_test, prediction4, target_names=['purchase_false', 'purchase_true']))


# model5: Random Foreast
rfc = RandomForestClassifier()
rfc_params = {'max_depth': (2,10,15), 
              'min_samples_split': range(2,5),
              'min_samples_leaf': range(1,5),
              'n_estimators': range(5,10),
}
gs = GridSearchCV(rfc, rfc_params)
gsm = gs.fit(X_train, Y_train)
rfc = gsm.best_estimator_
model5 = rfc.fit(X_train, Y_train) 

y_prob = model5.predict_proba(X_test)[:, 1]
roc(y_prob)
print('-----------------------------------------------------')
y_pred5 = model5.predict(X_test)
plot_confusion(y_pred5)
print('-----------------------------------------------------')
# evaluating the model
print("Training Accuracy: ", model5.score(X_train, Y_train))
print("Testing Accuracy: ", model5.score(X_test, Y_test))
print('-----------------------------------------------------')
print(classification_report(Y_test,y_pred5))
print('-----------------------------------------------------')
