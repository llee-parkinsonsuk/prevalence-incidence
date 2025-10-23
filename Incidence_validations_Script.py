#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[1]:


df_patient=pd.read_csv("Patient.txt",delimiter='\t')

df_patient.info()


#Read the Practice file 
df_practice=pd.read_csv("Practice.txt",delimiter='\t')

df_practice.info()


#In Gold patient file we dont have pracid so extracting from patid
#df2023_patient['patid'] = df2023_patient['patid'].astype(str)
#df2023_patient['pracid'] = df2023_patient['patid'].str[-5:]


df_patient['pracid'] = df_patient['pracid'].astype(str)
df_practice['pracid'] = df_practice['pracid'].astype(str)

# Now perform the merge using pracid
df_patient_practice = pd.merge(df_patient, df_practice, left_on='pracid', right_on='pracid')
df_patient_practice



# In[2]:


#The age should be atleast 20 in 2023,     
reference_year = 2023 - 20

#Dropping those whose age is less than 20 
df_patient_practice = df_patient_practice[df_patient_practice['yob'] <= reference_year]
df_patient_practice


# In[ ]:


#Registration Strart date Validation

#Patient should be registerd to the practice atleast six months before 02 july 2023, that is before 01 jan 2023


#Aurum the field is regstartdate
#Run this for validating Gold patients
#The date field is object convert to datetime
df_patient_practice['regstartdate'] = pd.to_datetime(df_patient_practice['regstartdate'], format='%d/%m/%Y')
cutoff_date = pd.Timestamp('2022-07-02')


df_patient_practice = df_patient_practice[df_patient_practice['regstartdate'].notnull() & (df_patient_practice['regstartdate'] < cutoff_date)]
df_patient_practice.info()


#Gold the field is crd
#Run the below for validating Gold patients

#The date field is object convert to datetime
#df_patient_practice['crd'] = pd.to_datetime(df_patient_practice['crd'], format='%d/%m/%Y')
#cutoff_date = pd.Timestamp('2023-01-01')


#df_patient_practice = df_patient_practice[df_patient_practice['crd'].notnull() & (df_patient_practice['crd'] < cutoff_date)]
#df_patient_practice.info()


# In[ ]:


#The Upto standaerd date should be before 01 jan 2023
#This UTS is only avalilbe in Gold so the validations is only for Gold patients

df_patient_practice['uts'] = pd.to_datetime(df_patient_practice['uts'], format='%d/%m/%Y')
cutoff_date = pd.Timestamp('2022-07-02')


df_patient_practice = df_patient_practice[df_patient_practice['uts'].notnull() & (df_patient_practice['uts'] < cutoff_date)]
df_patient_practice.info()


# In[4]:


#The Registraion end date should be after 02 july 2023 

###AURUM

# Convert tod from object type to datetime
df_patient_practice['regenddate'] = pd.to_datetime(df_patient_practice['regenddate'],format='%d/%m/%Y')

# Define the cutoff date
cutoff_date = pd.Timestamp('2022-12-31')


df_patient_practice = df_patient_practice[df_patient_practice['regenddate'].isnull() | (df_patient_practice['regenddate'] > cutoff_date)]
df_patient_practice


####GOLD
#In Gold the field name is tod(Transfer out date)
df_patient_practice['tod'] = pd.to_datetime(df_patient_practice['tod'],format='%d/%m/%Y')

# Define the cutoff date
cutoff_date = pd.Timestamp('2022-12-31')


df_patient_practice = df_patient_practice[df_patient_practice['tod'].isnull() | (df_patient_practice['tod'] > cutoff_date)]
df_patient_practice


# In[5]:


#Patient should be alive as of 02 july 2023

###AURUM

df_patient_practice['cprd_ddate'] = pd.to_datetime(df_patient_practice['cprd_ddate'],format='%d/%m/%Y')

# Define the cutoff date
cutoff_date = pd.Timestamp('2022-12-31')

# Filter the DataFrame to only include rows where death_date is null or death_date is after the cutoff date
df_patient_practice = df_patient_practice[df_patient_practice['cprd_ddate'].isnull() | (df_patient_practice['cprd_ddate'] > cutoff_date)]
df_patient_practice.info()

###GOLD
###In Gold the field name is deathdate


#df_patient_practice['deathdate'] = pd.to_datetime(df_patient_practice['deathdate'],format='%d/%m/%Y')

# Define the cutoff date
#cutoff_date = pd.Timestamp('2023-07-02')

# Filter the DataFrame to only include rows where death_date is null or death_date is after the cutoff date
#df_patient_practice = df_patient_practice[df_patient_practice['deathdate'].isnull() | (df_patient_practice['deathdate'] > cutoff_date)]
#df_patient_practice.info()





# In[6]:


#The latest collection date should be after 02 july 2023
##In both Gold and Aurum the field name is same

df_patient_practice['lcd'] = pd.to_datetime(df_patient_practice['lcd'],format='%d/%m/%Y')

# Define the cutoff date
cutoff_date = pd.Timestamp('2022-12-31')

# Filter the DataFrame to only include rows where lcd is after the cutoff date
df_patient_practice = df_patient_practice[df_patient_practice['lcd'] > cutoff_date]
df_patient_practice



# In[7]:


#The patients acceptable field should be 1

#Aurum
df_patient_practice = df_patient_practice[df_patient_practice['acceptable'] == 1]

#Gold the field name is accept
df_patient_practice = df_patient_practice[df_patient_practice['accept'] == 1]


# In[ ]:


###Afyter all these validatiuons the dataframe df_patient_practice contains only the validated patienst

