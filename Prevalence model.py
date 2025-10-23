#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[5]:


data=pd.read_csv('S:/Raisers Edge/IMPORTS/CPRD Data/Sandpit/Prasanth/Mice_imputeddata.txt',delimiter='\t')
data


# In[6]:


data.info()


# In[7]:


#Dropping the colum
data = data.drop(columns=['Unnamed: 0'])
data


# In[8]:


#Seperating the actual test cases from the rest in data1
#After fitting the model the model would be applied on this to get the final results
data1 = data[data['CATEGORY'] == 'No Diagnosis']
data1


# In[9]:


#data now contains only the cases hwree we know the diagnosis 
data = data[data['CATEGORY'] != 'No Diagnosis']
data


# In[10]:


data


# In[11]:


data['CATEGORY'].value_counts()


# In[12]:


#Applying stratify, this ensures the train-test split maintains the relative distribution of those variables.
data['stratify_col'] = data.apply(lambda x: f"{x['DATABASE']}_{x['GENDER']}_{x['SMOKINGSTATUS']}", axis=1)


# In[13]:


from sklearn.model_selection import train_test_split
data['stratify_col'] = data.apply(lambda x: f"{x['DATABASE']}_{x['GENDER']}_{x['SMOKINGSTATUS']}", axis=1)
# Assuming aurum_data is already loaded and preprocessed
train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['stratify_col'], random_state=42)

# Display the number of samples in each set
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")


# In[14]:


#Label encoding the target varibale, say here we are building binary model
#Here PD cases are reperested as 1 and others are 0
train_data['CATEGORY'] = np.where(train_data['CATEGORY'] == 'PD', 1, 0)
test_data['CATEGORY'] = np.where(test_data['CATEGORY'] == 'PD', 1, 0)


# In[15]:


train_data.info()


# In[16]:


##droping the patid and target variables, so this contains only the model features 
data_train_features=train_data.drop(["PATID","stratify_col","CATEGORY"],axis=1)
data_test_features=test_data.drop(["PATID","stratify_col","CATEGORY"],axis=1)
data_train_label=train_data["CATEGORY"]
data_test_label=test_data["CATEGORY"]


# In[18]:


#Fitting the random foirest model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_model.fit(data_train_features, data_train_label)


# In[19]:


data_train_features


# In[20]:


#Viewing the features importances in the model 
importances = rf_model.feature_importances_

# Get the feature names from the DataFrame
feature_names = data_train_features.columns

# Create a DataFrame to visualize the feature importances
import pandas as pd
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# In[21]:


feature_importance_df


# In[22]:


#Predicting on the testing set
data_test_predictions = rf_model.predict(data_test_features)


# In[23]:


#Checking the performance metrics
accuracy = accuracy_score(data_test_label, data_test_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(data_test_label, data_test_predictions))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(data_test_label, data_test_predictions))
cm = confusion_matrix(data_test_label, data_test_predictions)


# In[24]:


test_data


# In[26]:


#Plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
cm = confusion_matrix(data_test_label, data_test_predictions)
labels = ['NOT PD', 'PD']
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[27]:


data1.info()


# In[ ]:


#data1 have the actual No diagnosis cases where we need to apply the model and predict


# In[28]:


#Dropping the patid and target varibakle before applying the model
data1_features=data1.drop(["PATID","CATEGORY"],axis=1)


# In[29]:


data1_features


# In[30]:


#Predicting
predictions = rf_model.predict(data1_features)
predictions


# In[31]:


unique, counts = np.unique(predictions, return_counts=True)

# Creating a dictionary to show predictions count for each class
predictions_count = dict(zip(unique, counts))
print(predictions_count)


# In[32]:


predictions_list = predictions.tolist()  


# In[34]:


#To merge the predictions with patids
#We know 1 is PD and 0 is NOT PD
ids_list = data1['PATID'].tolist() 


# In[35]:


d = {'Id': ids_list, 'Predicted_Class': predictions_list}
result_df = pd.DataFrame(d)


# In[36]:


result_df


# In[ ]:


#Result_df contains the patinets now where Predicted_class=1 they are PD patienst predicted by the model

