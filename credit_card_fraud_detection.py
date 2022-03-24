#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 

# In[1]:


import pandas as pd
df=pd.read_csv('creditcard.csv')
df.head()


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df['Class'].value_counts()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


def draw_histograms(dataframe,features,rows,cols):
    fig=plt.figure(figsize=(20,20))
    for i,feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+"Distribution",color="DarkRed")
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()
draw_histograms(df,df.columns,8,4)


# In[12]:


### Independent and Dependent Features
X=df.drop("Class",axis=1)
y=df.Class


# # Sklearn Library installing

# In[13]:


pip install scikit-learn


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# In[15]:


log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75)


# In[17]:


clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(x_train,y_train)


# In[18]:


y_pred=clf.predict(x_test)


# In[19]:


# confusion matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu");


# In[20]:


print(accuracy_score(y_test,y_pred))


# In[21]:


print(classification_report(y_test,y_pred))


# # Random Forest classifier:

# In[22]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(criterion='gini',max_depth=10,min_samples_split=5,min_samples_leaf=1)
classifier.fit(x_train,y_train)


# In[23]:


y_pred=classifier.predict(x_test)


# In[24]:


# confusion matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu");


# In[25]:


print(accuracy_score(y_test,y_pred))


# In[26]:


print(classification_report(y_test,y_pred))


# # Importing imblearn library for performing under sampling:

# In[27]:


pip install imbalanced-learn


# # Performing  Under sampling:NearMiss

# In[28]:


from collections import Counter
Counter(y_train)


# In[29]:


from collections import Counter
from imblearn.under_sampling import NearMiss
ns=NearMiss(version=1,n_neighbors=3)
x_train_ns,y_train_ns=ns.fit_resample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[30]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train_ns,y_train_ns)


# In[31]:


y_pred=classifier.predict(x_test)


# In[32]:


# confusion matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu");


# In[33]:


print(accuracy_score(y_test,y_pred))


# In[34]:


print(classification_report(y_test,y_pred))


# # CatBoost:Overfit Detector

# In[35]:


pip install catboost


# In[36]:


# map categorical features
credit_catboost_ready_df=df.dropna()

features=[feat for feat in list(credit_catboost_ready_df) if feat !='Class']
print(features)
card_categories= np.where(credit_catboost_ready_df[features].dtypes !=np.float)[0]
card_categories


# In[64]:


SEED=1234
from catboost import CatBoostClassifier
 
params={'iterations':5000,
        'learning_rate':0.01,
        'cat_features':card_categories,
        'depth':3,
        'eval_metric':'AUC',
        'verbose':200,
        'od_type':"Iter",
        'od_wait':500,
        'random_seed':SEED
       }

cat_model = CatBoostClassifier(**params)
cat_model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True,plot=True);


# # AdaBoost Classsifier:

# # Adaboost library Installing

# In[38]:


pip install ada-boost


# In[39]:


RANDOM_STATE = 2018
NUM_ESTIMATORS = 100
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


# In[40]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                          n_estimators=NUM_ESTIMATORS)


# In[41]:


clf.fit(df[predictors],df['Class'].values)


# In[42]:


y_pred = clf.predict(df[predictors])


# In[43]:


cm=pd.crosstab(df[target].values,y_pred,rownames=['Actual'],colnames=['Predicted'])
fig, (ax1)=plt.subplots(ncols=1,figsize=(5,5))
sns.heatmap(cm,
           xticklabels=['Not Fraud','Fraud'],
           yticklabels=['Not Fraud','Fraud'],
           annot=True,ax=ax1,
           linewidths=2,linecolor="Darkblue",cmap="Blues")
plt.title('Confusion Matrix',fontsize=14)
plt.show()


# # Installing matplotlib for plotting graph

# In[44]:


pip install matplotlib 


# In[45]:


pip install seaborn


# In[46]:


# confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = cat_model.predict(x_test)


# In[47]:


# confusion matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu");


# # SMOTE Analysis

# In[48]:


pip install imblearn


# In[49]:


from imblearn.combine import SMOTETomek


# In[50]:


os=SMOTETomek(random_state=42)
x_train_ns,y_train_ns=os.fit_resample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[51]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train_ns,y_train_ns)


# In[65]:


y_pred=classifier.predict(x_test)


# In[54]:


# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu");


# In[55]:


print(accuracy_score(y_test,y_pred))


# In[56]:


print(classification_report(y_test,y_pred))


# In[57]:


df_temp=df.drop(columns=['Time','Amount','Class'],axis=1)

# create dist plots
fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20,50))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df_temp[col], ax=ax[index])
    index +=1
plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=5)


# In[59]:


legit=df[df.Class==0]
fraud=df[df.Class==1]


# In[60]:


print(legit.shape)
print(fraud.shape)


# In[61]:


legit.Amount.describe()


# In[62]:


fraud.Amount.describe()


# In[63]:


df.groupby('Class').mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




