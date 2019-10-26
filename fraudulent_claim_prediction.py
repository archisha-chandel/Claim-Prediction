#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                           # removing all the ouliers and missing values 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 7
import time
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[167]:


train = pd.read_csv(r'C:\Users\welcome\Downloads\file\train.csv')
test = pd.read_csv(r'C:\Users\welcome\Downloads\file\test.csv')


# In[168]:


train.head()


# In[169]:


# Inspect train set
print ("Length of train rows:", len(train))
inspect_train = pd.DataFrame({'Dtype': train.dtypes, 'Unique values': train.nunique() ,
             'Number of Missing values': train.isnull().sum() ,
              'Percentage Missing': (train.isnull().sum() / len(train)) * 100
             }).sort_values(by='Number of Missing values',ascending = False)
inspect_train


# In[170]:


# Inspect test set
print ("Length of test rows:", len(test))
inspect_test = pd.DataFrame({'Dtype': test.dtypes, 'Unique values': test.nunique() ,
             'Number of Missing values': test.isnull().sum() ,
              'Percentage Missing': (test.isnull().sum() / len(test)) * 100
             }).sort_values(by='Number of Missing values',ascending = False)
inspect_test


# In[171]:


test.columns


# ## EDA preprocessing

# In[172]:


train.drop(['ID','Gender'],axis = 1,inplace=True)


# In[173]:


train[train['Duration']<0]


# In[174]:


train.drop(train.index[[4063,38935,48367]], inplace=True)


# In[175]:


train[train['Age']==0]


# In[176]:


train.drop(train.index[[1362,45828]],axis = 0,inplace = True)


# In[177]:


plt.hist(train['Age'])


# In[178]:


train[train['Age']==118]


# In[179]:


train.shape


# In[180]:


792/50548


# In[181]:


selRows = train[train['Age']==118].index
train = train.drop(selRows, axis=0)


# In[182]:


train[train['Age']==118]


# In[183]:


train.shape


# In[184]:


50553-49756            # original train - deleted train


# In[185]:


797/50553   #percent of removed values


# In[186]:


train.describe()


# In[187]:


plt.hist(train['Age'])


# In[188]:


print('Percentage Missing: ' , (train.isnull().sum() / len(train)) * 100)


# In[189]:


train_x=train.drop("Claim",axis=1)
y = train['Claim']


# In[ ]:





# In[190]:


def detect_outliers(dataframe):
    cols = list(dataframe)
    
    for column in cols:
        if column in dataframe.select_dtypes(include=np.number).columns:
            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)
            iqr = q3 - q1
            fence_low = q1 - (1.5*iqr)
            fence_high = q3 + (1.5*iqr)

            print(column + ' ---------', dataframe.loc[(dataframe[column] < fence_low) | (dataframe[column] > fence_high)].shape[0])

detect_outliers(train_x)


# In[191]:


from scipy.stats.mstats import winsorize
def treat_outliers(dataframe):
    cols = list(dataframe)
    for col in cols:
        if col in dataframe.select_dtypes(include=np.number).columns:
            dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))
    
    return dataframe    


train_x = treat_outliers(train_x)
print(detect_outliers(train_x))


# In[192]:


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = 'viridis',
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    
    correlation = df.corr().unstack().sort_values(kind='quicksort')
    print('Highly Correlated Variables')
    return correlation[((correlation>=0.75) | (correlation<=-0.75)) & (correlation!=1)]

correlation_heatmap(train)


# In[193]:


def class_imbalance(target):
    class_values = (target.value_counts()/target.value_counts().sum())*100
    return class_values

class_imbalance(y)


# In[194]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Function that auto encodes any dataframe column of type category or object.
def dummyEncode(dataset):
        
        columnsToEncode = list(dataset.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                dataset[feature] = le.fit_transform(dataset[feature])
            except:
                print('Error encoding '+feature)
        return dataset
train_x = dummyEncode(train_x)
y = pd.DataFrame(le.fit_transform(y))


# # Baseline Model
#  
#   We can use a logistic regression in order to see baseline performance on this problem.
#   

# ## Task 1 : Fit a vanilla Logistic Regression model on the training set and predict on the test set and plot the confusion matrix, accuracy, precision_score for the predicted model

# In[195]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# In[196]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix


# In[197]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_x,y,test_size=0.2,random_state=42)


# In[198]:


clf = LogisticRegression().fit(X_train, Y_train)

Y_test_pred = clf.predict(X_test)
target = train['Claim']
print("Accuracy = " , accuracy_score(Y_test, Y_test_pred))
print("Precision = " ,precision_score(Y_test, Y_test_pred))
print("Recall = " ,recall_score(Y_test, Y_test_pred))
#print("F1 Score = " ,f1_score(Y_test, Y_test_pred))

confusion_matrix(Y_test, Y_test_pred)


# #### The precision is low and this is because from the matrix above, we can see that the False Negatives are too many in the data. This was supposed to happen because since the target is highly imbalanced with lesser number of 1s, our model has learnt to predict only 0s most of the times. Hence we have to treat this class imbalance first

# ## Task 2 : Set the parameter class_weight=balanced inside Logistic Regression and check for the metrics calculated above and also the confusion matrix

# In[199]:


clf = LogisticRegression(class_weight='balanced').fit(X_train, Y_train)

Y_test_pred_balanced = clf.predict(X_test)

print("Accuracy = " , accuracy_score(Y_test, Y_test_pred_balanced))
print("Precision = " ,precision_score(Y_test, Y_test_pred_balanced))
print("Recall = " ,recall_score(Y_test, Y_test_pred_balanced))
#print("F1 Score = " ,f1_score(Y_test, Y_test_pred_balanced))
confusion_matrix(Y_test, Y_test_pred_balanced)


# ## Task 3 : Perform Random Undersampling on the train data and then fit a Logistic regression model on this undersampled data and then predict on the test data and calculate the precision and the confusion matrix.

# In[200]:


get_ipython().system('pip install imblearn')


# In[201]:


from imblearn.under_sampling import RandomUnderSampler
#Code starts here

# Create random under sampler object
rus = RandomUnderSampler(random_state=0)

#Undersampling the train data
X_sample_1, y_sample_1 =  rus.fit_sample(X_train, Y_train)


#Initiating a logistic regression model
model_rus = LogisticRegression()

#Fitting the model with sampled data
model_rus.fit(X_sample_1, y_sample_1)

#Making prediction of test values
Y_pred=model_rus.predict(X_test)

# Calculating the necessary metrics
print("Accuracy = " , accuracy_score(Y_test, Y_pred))
print("Precision = " ,precision_score(Y_test, Y_pred))
print("Recall = " ,recall_score(Y_test, Y_pred))
#print("F1 Score = " ,f1_score(Y_test, Y_pred))
confusion_matrix(Y_test, Y_pred)


# ## Task 4 : Perform Tomek Undersampling on the train data and then fit a Logistic regression model on this undersampled data and then predict on the test data and calculate the precision_score, the confusion matrix.

# In[202]:


from imblearn.under_sampling import TomekLinks

#Code starts here

#Initialising Tomek Links object
tl = TomekLinks()

#Undersamlpling the train data
X_sample4, y_sample4 = tl.fit_sample(X_train, Y_train)

# Plot the distribution of the target using a countplot
sns.countplot(y_sample4)

#Initialising the logistic regression model
model_tl = LogisticRegression()

#Fitting the model with sampled data
model_tl.fit(X_sample4, y_sample4)

#Making the predictions with test data
Y_pred_tomek=model_tl.predict(X_test)

# Calculating the necessary metrics
print("Accuracy = " , accuracy_score(Y_test, Y_pred_tomek))
print("Precision = " ,precision_score(Y_test, Y_pred_tomek))
print("Recall = " ,recall_score(Y_test, Y_pred_tomek))
confusion_matrix(Y_test, Y_pred_tomek)


# #### Tomek Undersampling doesn't seem a good fit for data. There is hardly any increase in recall compared to the vanilla model. Undersampling techniques, even if they provide an increase in the metric of choice, are not favoured since you tend to lose some information when you undersample the majority class of the target. Hence in most cases, what we prefer to perform are Oversampling techniques like Random Oversampling and SMOTE

# In[203]:



from imblearn.over_sampling import RandomOverSampler

#Code starts here

#Initialise the random over sampler object
ros = RandomOverSampler(random_state=0)

#Sample the train data using random over sampling method
X_sample_2, y_sample_2 = ros.fit_sample(X_train, Y_train)

# Using a countplot 
sns.countplot(y_sample_2)

#Initialising a logsitic regression model
model_ros = LogisticRegression()

#Fitting the model with train data
model_ros.fit(X_sample_2, y_sample_2)

#Making predictions of the train data
Y_prediction=model_ros.predict(X_test)

# Calculating the necessary metrics
print("Accuracy = " , accuracy_score(Y_test, Y_prediction))
print("Precision = " ,precision_score(Y_test, Y_prediction))
print("Recall = " ,recall_score(Y_test, Y_prediction))
#print("F1 Score = " ,f1_score(Y_test, Y_pred))
confusion_matrix(Y_test, Y_prediction)
#Finding the confusion matrix 
#pd.crosstab(Y_pred, Y_test[target], rownames=['Predicted'], colnames=['Actual'])


# #### So as you can observe from the above plot, oversampling has brought an equal balance in the distribution of classes in the target variable. Also the precision is much better compared to the vanilla model and Tomek undersampling 

# ## Task 5 : Perform SMOTE on the train data and then fit a Logistic regression model on this undersampled data and then predict on the test data and calculate the precision, recall, accuracy, f1-score and the confusion matrix.

# In[204]:


from imblearn.over_sampling import SMOTE


#Code starts here

#Initialising a SMOTE object
smote = SMOTE(random_state=12,ratio=1.0)

#Sampling the data using SMOTE
X_sample_3, y_sample_3 = smote.fit_sample(X_train, Y_train)

# Using a countplot plot the distribution of y_sample_3
sns.countplot(y_sample_3)

#Initialising Logistic Regression model
model_smote = LogisticRegression()

#Fitting the model on train data
model_smote.fit(X_sample_3, y_sample_3)

#Making predictions on test data
Y_pred_smote=model_smote.predict(X_test)

#Finding the accuracy score 
accuracy_smote=model_smote.score(X_test,Y_test)
print("Accuracy:",accuracy_smote)       


#Finding the recall score
recall_smote=recall_score(Y_test, Y_pred_smote)
print ("recall:",recall_smote)

#Finding the precision score
precision_smote=precision_score(Y_test, Y_pred_smote)
print ("precision:",precision_smote)
confusion_matrix(Y_test, Y_pred_smote)               # 1926 are predicted 0 but are actually 1 because data contains mostly 0
                                                    # this is the reason precision is low


# #### SMOTE IS GIVING BETTER PRECISION THAN ANY OTHER TECHNIQUES.
This poor performance by the logistic regression indicates the problem of separating true claim is non-linear.
# ## More Complex Model
For a potentially better machine learning model, we can move to the Random Forest Classifier. From the results of the logistic regression, this looks to be a non-linear problem which means we should use a model capable of learning a non-linear decision boundary.

We'll use most of the default hyperparameters but alter a few to prevent overfitting. We can also set class_weight = 'balanced' to try and offset the impact of such an imbalanced classification problem.
# In[205]:


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
#from sklearn.linear_model import LogisticRegression

#model = LogisticRegression(random_state=50)
X_train_m,X_test_m,Y_train_m,Y_test_m = train_test_split(train_x,y,test_size=0.2,random_state=42)
def evaluate(model, train, y, test, test_y):
    """Evaluate a machine learning model on four metrics:
       ROC AUC, precision score, recall score, and f1 score.

       Returns the model and the predictions."""

    model.fit(train, y)

    # Predict probabilities and labels
    probs = model.predict_proba(test)[:, 1]
    preds = model.predict(test)

    # Calculate ROC AUC
    roc = roc_auc_score(test_y, probs)
    name = repr(model).split('(')[0]
    print(f"{name}\n")
    print(f'ROC AUC: {round(roc, 4)}')

    # Iterate through metrics
    for metric in [precision_score, recall_score, f1_score]:
        # Use .__name__ attribute to list metric
        print(f'{metric.__name__}: {round(metric(test_y, preds), 4)}')

    return model, preds


model, preds = evaluate(model, X_train, Y_train_m, X_test_m, Y_test_m)


# In[206]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=40,
                               min_samples_leaf=50,
                               n_jobs=-1, class_weight='balanced',
                               random_state=50)

model, preds = evaluate(model, X_train, Y_train_m, X_test_m, Y_test_m)

The random forest performance is much better than just guessing! With no tuning, the model is able to identify 36% of the Claims and the false positives have been reduced.
# ## Using SMOTE for random forest 

# In[207]:


from imblearn.over_sampling import SMOTE


#Code starts here

#Initialising a SMOTE object
smote = SMOTE(random_state=12,ratio=1.0)

#Sampling the data using SMOTE
X_sample_3, y_sample_3 = smote.fit_sample(X_train, Y_train)

# Using a countplot plot the distribution of y_sample_3
sns.countplot(y_sample_3)

#Initialising Logistic Regression model
model_smote = RandomForestClassifier(n_estimators=100, max_depth=40,
                               min_samples_leaf=50,
                               n_jobs=-1, class_weight='balanced',
                               random_state=50)

#Fitting the model on train data
model_smote.fit(X_sample_3, y_sample_3)

#Making predictions on test data
Y_pred=model_smote.predict(X_test)

#Finding the accuracy score 
accuracy_smote=model_smote.score(X_test,Y_test)
print("Accuracy:",accuracy_smote)       


#Finding the recall score
recall_smote=recall_score(Y_test, Y_pred)
print ("recall:",recall_smote)

#Finding the precision score
precision_smote=precision_score(Y_test, Y_pred)
print ("precision:",precision_smote)


# In[208]:


from sklearn.metrics import precision_recall_curve, confusion_matrix

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn')


def plot_precision_recall(test_y, probs, title='Precision Recall Curve', threshold_selected=None):
    """Plot a precision recall curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    precision, recall, threshold = precision_recall_curve(Y_test_m, probs)
    plt.figure(figsize=(10, 8))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall', size=18)
    plt.ylabel('Precision', size=18)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, size=20)
    plt.xticks(size=14)
    plt.yticks(size=14)

    if threshold_selected:
        p = precision(np.where(threshold == threshold_selected)[0])
        r = recall(np.where(threshold == threshold_selected)[0])
        plt.scatter(r, p, marker='*', size=200)
        plt.vlines(r, ymin=0, ymax=p, linestyles='--')
        plt.hlines(p, xmin=0, xmax=r, linestyles='--')

    pr = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1],
                       'threshold': threshold})
    return pr


probs = model.predict_proba(X_test_m)[:, 1]
pr_data = plot_precision_recall(
    Y_test_m, probs, title='Precision-Recall Curve for Random Forest')


# In[209]:


precision_above = pr_data.loc[pr_data['precision'] >= 0.2].copy()
precision_above.sort_values('recall', ascending=False, inplace=True)
precision_above.head()

We can see that if we want a precision of 22%, then our recall will be 2.9%. This means we'll miss over 97% of the true Claims in the data 
This means that in order to identify 2.9% of the actual claims, we'll have to accept that only 22.22% of the predicted positives are actually positive claims.
# ## Adjusting for the Business Requirement 
Let's say we are required to have a recall of 50% in our model. This means our model finds 50% of the true claims in the data. We'll work through the rest of this notebook under this assumption. To find the threshold, we use:
# In[210]:


recall_attained = 0.5
recall_above = pr_data.loc[pr_data['recall'] >= recall_attained].copy()
recall_above.sort_values('precision', ascending=False, inplace=True)        
recall_above.head()

By increasing the threshold we can increase the precision but model will make no sense as it only recognise 50% of the actual positive claims and it doesnt say much about rest 50% actual positive claims.so , remaining 50% customers can sue us for wrongly denying the claims although it is true.so only increasing the precision wont help us we have to balance both , i.e precision and recall
# In[211]:


precision_attained = recall_above.iloc[0, 0]
threshold_required = recall_above.iloc[0, -1]

print(
    f'At a threshold of {round(threshold_required, 4)} the recall is {100 * recall_attained:.2f}% and the precision is {round(100 * precision_attained, 4)}%')

so we have to find a perfect trade off between precision and recall ,so as to save my company of getting sued by wrongly denying the claims
While better precision is good, it might be coming at the expense of a large reduction in recall. In general, we need to look at both precision and recall together, or summary metrics like AUC
# ## Precision Recall Curve 
One of the best methods for tuning a model for a business need is through the precision recall curve. This shows the precision-recall tradeoff for different thresholds. Depending on the business requirement, we can change the threshold for classifying a positive example to alter the balance of true positives, false positives, false negatives, and true negatives. There will always be a tradeoff between precision and recall, but we can try to find the right balance by visually and quantitatively assessing the model.


# In[212]:


def plot_precision_recall(test_y, probs, title='Precision Recall Curve', threshold_selected=None):
    """Plot a precision recall curve for predictions. 
       Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py"""

    precision, recall, threshold = precision_recall_curve(Y_test_m, probs)
    plt.figure(figsize=(10, 10))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall', size=24)
    plt.ylabel('Precision', size=24)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, size=24)
    plt.xticks(size=18)
    plt.yticks(size=18)

    if threshold_selected:
        p = precision[np.where(threshold == threshold_selected)[0]]
        r = recall[np.where(threshold == threshold_selected)[0]]
        plt.scatter(r, p, marker='*', s=600, c='r')
        plt.vlines(r, ymin=0, ymax=p, linestyles='--')
        plt.hlines(p, xmin=0, xmax=r, linestyles='--')
        plt.text(r - 0.1, p + 0.15,
                 s=f'Threshold: {round(threshold_selected, 2)}', size=20, fontdict={'weight': 1000})
        plt.text(r - 0.2, p + 0.075,
                 s=f'Precision: {round(100 * p[0], 2)}% Recall: {round(100 * r[0], 2)}%', size=20,
                 fontdict={'weight': 1000})

    pr = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1],
                       'threshold': threshold})
    return pr


# In[213]:


pr_data = plot_precision_recall(Y_test_m, probs, title='Precision-Recall Curve for Tuned Random Forest',
                                threshold_selected=threshold_required)


# In[214]:


from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.style.use('bmh')
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=20)
    plt.yticks(tick_marks, classes, size=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=20)
    plt.grid(None)
    plt.ylabel('True label', size=22)
    plt.xlabel('Predicted label', size=22)
    plt.tight_layout()


# In[215]:


# Make predictions where probability is above threshold
preds = np.zeros(len(Y_test))
preds[probs >= threshold_required] = 1

# Make and plot confusion matrix
cm = confusion_matrix(Y_test, preds)
plot_confusion_matrix(cm, classes=['No Claim', 'Claim'],
                      title='Churn Confusion Matrix')

If we satisfy our business requirement, this is the best prediction of what our performance would be on new data. The model is able to identiy 50% of churned customers compared to a baseline of around 1%. The precision has increased from the baseline 0% to 8%, a relative increase of over 800%.
# ## Feature Importances 

# In[216]:


fi = pd.DataFrame({'importance': model.feature_importances_}, index=train_x.columns).    sort_values('importance', ascending=False)
fi


# ## Decision tree classifier 

# In[217]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier( max_depth=40,
                               min_samples_leaf=50,
                                class_weight='balanced',
                               random_state=50)

model, preds = evaluate(model, X_train, Y_train_m, X_test_m, Y_test_m)


# In[ ]:


With no tuning, the model is able to identify 36% of the Claims and the false positives have been reduced.

