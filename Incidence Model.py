#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snowflake.connector
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


# In[2]:


##Connecting snowflake and reading the data table
credentials={
    'account':'parkinson.eu-west-1',
    'user':'pairuthayaraj@parkinsons.org.uk',
    'authenticator':'externalbrowser',
    
}

con = snowflake.connector.connect(
    user="pairuthayaraj@parkinsons.org.uk", 
    account="parkinson.eu-west-1", 
    authenticator="externalbrowser",
    role="TEAM_DATA_ANALYSTS",
    warehouse="DEV_WH",
    database="SANDPIT",
    schema="PAIRUTHAYARAJ"
)


# In[34]:


#Read Aurum model data
Aurum_data=pd.read_sql("SELECT * FROM SANDPIT.PAIRUTHAYARAJ.AURUM_INCIDENCE_MODEL_VARIABLES",con)
Aurum_data.head()


# In[35]:


Aurum_data['DIAGNOSIS'].value_counts()


# In[36]:


#Read Gold model data
Gold_data=pd.read_sql("SELECT * FROM SANDPIT.PAIRUTHAYARAJ.GOLD_INCIDENCE_MODEL_VARIABLES",con)
Gold_data.head()


# In[37]:


Gold_data['DIAGNOSIS'].value_counts()


# In[38]:


Aurum_data


# In[39]:


#Only BMI has some null values at this point in both Aurum and Gold


# In[40]:


Aurum_data['BMI'] = pd.to_numeric(Aurum_data['BMI'], errors='coerce')

# Calculate the mean BMI, ignoring NaN values and values less than or equal to 12 or greater than 60
median_bmi = Aurum_data[(Aurum_data['BMI'] > 12) & (Aurum_data['BMI'] <= 65)]['BMI'].median()

# Replace null values, values less than or equal to 12, and values greater than 60 with the mean BMI
Aurum_data['BMI'].fillna(median_bmi, inplace=True)
Aurum_data.loc[(Aurum_data['BMI'] <= 12) | (Aurum_data['BMI'] > 65), 'BMI'] = median_bmi


# In[41]:


Gold_data['BMI'] = pd.to_numeric(Gold_data['BMI'], errors='coerce')

# Calculate the mean BMI, ignoring NaN values and values less than or equal to 12 or greater than 60
median_bmi = Gold_data[(Gold_data['BMI'] > 12) & (Gold_data['BMI'] <= 65)]['BMI'].median()

# Replace null values, values less than or equal to 12, and values greater than 60 with the mean BMI
Gold_data['BMI'].fillna(median_bmi, inplace=True)
Gold_data.loc[(Gold_data['BMI'] <= 12) | (Gold_data['BMI'] > 65), 'BMI'] = median_bmi


# In[42]:


Aurum_data.info()


# In[43]:


Gold_data.info()


# In[44]:


data = pd.concat([Gold_data, Aurum_data], ignore_index=True)
data


# In[45]:


data


# In[46]:


data.info()


# In[47]:


#Copying the actual test cases to data1 to do the prediction at the last 
data1 = data[data['DIAGNOSIS'] == 'test']
data1


# In[48]:


#Dropping the test cases to bukd the model for now 
data = data[data['DIAGNOSIS'] != 'test']
data


# In[49]:


data.info()


# In[50]:


#Stratify to do the TRAIN AND test split
data['stratify_col'] = data.apply(lambda x: f"{x['DATASOURCE']}_{x['GENDER']}_{x['SMOKINGSTATUS']}", axis=1)


# In[51]:


from sklearn.model_selection import train_test_split
data['stratify_col'] = data.apply(lambda x: f"{x['DATASOURCE']}_{x['GENDER']}", axis=1)

train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['stratify_col'], random_state=42)

# Display the number of samples in each set
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")


# In[52]:


#Label encoding 
label_encoders = {}
categorical_columns = ['SMOKINGSTATUS', 'GENDER',  'DATASOURCE','REGION']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])


# In[53]:


#Label encoding 
label_encoders = {}
categorical_columns = ['SMOKINGSTATUS', 'GENDER',  'DATASOURCE','REGION']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    test_data[column] = label_encoders[column].fit_transform(test_data[column])


# In[54]:


train_data['DIAGNOSIS'].value_counts()


# In[55]:


#Three categories Model (PD,NO Diagnosis,Others)
#train_data['DIAGNOSIS'] = train_data['DIAGNOSIS'].apply(
#    lambda x: 1 if x == 'pd' else (0 if x == 'No diagnosis' else 2)
#)


# In[56]:


#Binary Model(PD,NOT PD)
train_data['DIAGNOSIS'] = train_data['DIAGNOSIS'].apply(lambda x: 1 if x == 'pd' else 0)


# In[57]:


train_data['DIAGNOSIS'].value_counts()


# In[58]:


#Binary Model(PD,NOT PD)
test_data['DIAGNOSIS'] = test_data['DIAGNOSIS'].apply(lambda x: 1 if x == 'pd' else 0)


# In[59]:


train_data.info()


# In[60]:


data_train_features=train_data.drop(["patid","stratify_col","DIAGNOSIS"],axis=1)
data_test_features=test_data.drop(["patid","stratify_col","DIAGNOSIS"],axis=1)
data_train_label=train_data["DIAGNOSIS"]
data_test_label=test_data["DIAGNOSIS"]


# In[61]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[62]:


#Random Forest Model


# In[63]:


rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the oversampled training data
rf_model.fit(data_train_features, data_train_label)


# In[64]:


data_test_predictions = rf_model.predict(data_test_features)


# In[65]:


accuracy = accuracy_score(data_test_label, data_test_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(data_test_label, data_test_predictions))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(data_test_label, data_test_predictions))
cm = confusion_matrix(data_test_label, data_test_predictions)


# In[ ]:


##Applying the model to the actual test cases


# In[68]:


#Label Encoding
label_encoders = {}
categorical_columns = ['SMOKINGSTATUS', 'GENDER', 'REGION', 'DATASOURCE']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data1[column] = label_encoders[column].fit_transform(data1[column])


# In[69]:


data1_features = data1.drop(columns=['patid','DIAGNOSIS'])


# In[70]:


data1


# In[71]:


#predcit using rf model
predictions = rf_model.predict(data1_features)
predictions
unique, counts = np.unique(predictions, return_counts=True)

# Creating a dictionary to show predictions count for each class
predictions_count = dict(zip(unique, counts))
print(predictions_count)


# In[72]:


#Mapping the predictions to patids
predictions_list = predictions.tolist()  
ids_list = data1['patid'].tolist() 
d = {'Id': ids_list, 'Predicted_Class': predictions_list}
result_df = pd.DataFrame(d)


# In[73]:


result_df

