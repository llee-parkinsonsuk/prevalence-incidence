#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd


# In[39]:


#Loading the medcodes we are intrested in say MSA,PSP,CBS,DLB,...PD 
dgcodes=pd.read_csv('medcodes.csv')
dgcodes


# In[40]:


#get the patients obseravtion for whom we need to run the ruleset
#In the case of Gold need to get the clinical file
obs=pd.read_csv('observation.txt',delimiter='\t')
obs


# In[43]:


#in this case we need to run the ruleset for secular trend 2017 
patients=pd.read_csv('validated_patients_for2017 ',delimiter='\t')
patients


# In[44]:


#getting observation for only those validated  patients
obs1=obs[obs['patid'].isin(patients['patid'])]
obs1


# In[46]:


#Convert to datetime
#some cases obsdate is null so use enterdate instead of obsdate 
obs1['obsdate'] = pd.to_datetime(obs1['obsdate'], format='%d/%m/%Y')
obs1['enterdate'] = pd.to_datetime(obs1['enterdate'], format='%d/%m/%Y')
obs1['obsdate'] = obs1['obsdate'].fillna(obs1['enterdate'])


# In[47]:


#We would need only these fields to further analysis
obs1=obs1[['patid','obsdate','medcodeid']]
obs1


# In[48]:


#for 2017 prevelance we dont consider the future records beyond july 2017, so removing them
cutoff_date = pd.to_datetime('2017-07-02')

# Filter rows where obsdate is on or before the cutoff date
obs1 = obs1[obs1['obsdate'] <= cutoff_date]
obs1


# In[51]:


dgcodes=dgcodes[['medcodeid','diagnostic group']]
dgcodes


# In[52]:


dgcodes['diagnostic group'].unique()


# In[53]:


obs1['medcodeid']=obs1['medcodeid'].astype(str)


# In[54]:


obs1


# In[55]:


obs1['obsdate'] = pd.to_datetime(obs1['obsdate'])
obs1


# In[56]:


dgcodes.info()
dgcodes['medcodeid']=dgcodes['medcodeid'].astype(str)


# In[57]:


#with the observations merge the diagnostic group
obs1 = pd.merge(obs1, dgcodes, on='medcodeid', how='left')
obs1


# In[58]:


obs1.info()


# In[59]:


obs1=obs1[['patid','obsdate','diagnostic group']]


# In[60]:


obs1


# In[61]:


#obs = obs.dropna(subset=['diagnostic group'])

# Rename the 'diagnostic_group' column to 'Group'
#Done renbame as in the ruleset the code is developed to accept as Group colum name
obs1.rename(columns={'diagnostic group': 'Group'}, inplace=True)


# In[62]:


obs1


# In[63]:


obs1


# In[64]:


obs1['Group'].unique()


# In[65]:


#Add hes
#For all the validated patienst we need to also look into their HES admissions
hesobs=pd.read_csv('H:/CPRD Analysis/Prevalence 2023 Final Patients/Prevelance trend_otherparkinsons/Aurum/heshistory.csv')
hesobs


# In[67]:


#IN cpord thats obsdate so to merge change the eventdate into obsdate 
hesobs['obsdate'] = pd.to_datetime(hesobs['EVENTDATE'], format='%d/%m/%Y')
hesobs.drop(columns=['EVENTDATE'], inplace=True)

# Rename CATEGORY to Group
hesobs.rename(columns={'CATEGORY': 'Group'}, inplace=True)

# Replace values in Group column
hesobs['Group'] = hesobs['Group'].replace({'Parkinsons': 'PD', 'CBD': 'CBS'})


# In[68]:


hesobs.rename(columns={'PATID': 'patid'}, inplace=True)
hesobs=hesobs[['patid','obsdate','Group']]
hesobs


# In[69]:


patients


# In[70]:


#Hesobs have records of lal the patienst for all the years now we would need only the validatred patients of 2017 
hesobs=hesobs[hesobs['patid'].isin(patients['patid'])]
hesobs


# In[71]:


hesobs['patid'].nunique()


# In[72]:


#For 2017 we connsider only their records till July 2017
cutoff_date = pd.to_datetime('2017-07-02')

# Filter rows where obsdate is on or before the cutoff date
hesobs = hesobs[hesobs['obsdate'] <= cutoff_date]
hesobs


# In[74]:


#Merging patients CPRD and HES records
obs1=pd.concat([obs1, hesobs], ignore_index=True)
obs1


# In[76]:


#Apply the ruleset 


# In[77]:


# Conditions
all_conditions = ['MSA', 'PSP', 'CBS', 'DLB', 'VP', 'DIP', 'Secondary','PD']
degen_conditions = ['MSA', 'PSP', 'CBS']
nondegen_conditions = ['VP', 'DIP', 'Secondary']
pd_condition = 'PD'
dlb_condition = 'DLB'
vp_condition = 'VP'
dip_condition = 'DIP'
sec_condition = 'Secondary'

# Lists to hold patients
pd_list = []
msa_list = []
psp_list = []
cbs_list = []
dlb_list = []
vp_list = []
dip_list = []
sec_list = []

# Counter for patients passing the first if condition
patients_pass_first_condition = 0
gc_in_condition_6_count = 0

# Helper functions
def get_most_recent_group(patient_records):
    recent_obs = patient_records.loc[patient_records['obsdate'].idxmax()]
    return recent_obs['Group']

def is_dlb_within_one_year(pd_date, dlb_date):
    return (dlb_date - pd_date).days > 365

#debug_patid = 11879384621145

# Iterate over unique patients
for patid in obs1['patid'].unique():
    patient_records = obs1[obs1['patid'] == patid]
    groups = patient_records['Group'].unique()
    
    # Collecting group status
    has_pd = pd_condition in groups
    has_dlb = dlb_condition in groups
    has_vp = vp_condition in groups
    has_dip = dip_condition in groups
    has_sec = sec_condition in groups
    has_degen = any(group in degen_conditions for group in groups)
    has_nondegen = any(group in nondegen_conditions for group in groups)

    # Filter patients based on the main conditions
    if any(group in all_conditions for group in groups):
        # Increment the counter for patients passing the first if condition
        patients_pass_first_condition += 1

        if has_dlb:
            if not has_vp and not has_dip and not has_degen and not has_pd and not has_sec:
                dlb_list.append(patid)
                continue
        
        if has_vp:
            if not has_dlb and not has_dip and not has_degen and not has_pd and not has_sec:
                vp_list.append(patid)
                continue
        
        if has_dip:
            if not has_dlb and not has_vp and not has_degen and not has_pd and not has_sec:
                dip_list.append(patid)
                continue
        
        if has_sec:
            if not has_dlb and not has_vp and not has_dip and not has_pd and not has_degen:
                sec_list.append(patid)
                continue
        
        if has_degen:
            if not has_dlb and not has_vp and not has_dip and not has_sec and not has_pd:
                #if patid == debug_patid:
                   # print(f"Patient {patid} has degen conditions but not DLB, VP, DIP, SEC, or PD.")
                filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS'])]
                recent_group = get_most_recent_group(filtered_records)
                if recent_group == 'MSA':
                    msa_list.append(patid)
                elif recent_group == 'PSP':
                    psp_list.append(patid)
                elif recent_group == 'CBS':
                    cbs_list.append(patid)
                continue
              
        if has_degen and has_pd:
            if not has_dlb and not has_vp and not has_dip and not has_sec:
                filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS', 'PD'])]
                recent_group = get_most_recent_group(filtered_records)
                
                if recent_group == 'MSA':
                    msa_list.append(patid)
                elif recent_group == 'PSP':
                    psp_list.append(patid)
                elif recent_group == 'CBS':
                    cbs_list.append(patid)
                elif recent_group == 'PD':
                    pd_list.append(patid)    
                continue
        
        if has_degen and has_pd and has_dlb:
            pd_obsdate = patient_records[patient_records['Group'] == 'PD']['obsdate'].min()
            dlb_obsdate = patient_records[patient_records['Group'] == 'DLB']['obsdate'].min()
            if is_dlb_within_one_year(pd_obsdate, dlb_obsdate):
                pd_list.append(patid)
            else:
                filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS', 'DLB'])]
                recent_group = get_most_recent_group(filtered_records)
                if recent_group == 'MSA':
                    msa_list.append(patid)
                elif recent_group == 'PSP':
                    psp_list.append(patid)
                elif recent_group == 'CBS':
                    cbs_list.append(patid)
                else:
                    dlb_list.append(patid)
            continue    
                 
        if has_dlb and has_pd and not has_degen:
            pd_obsdate = patient_records[patient_records['Group'] == 'PD']['obsdate'].min()
            dlb_obsdate = patient_records[patient_records['Group'] == 'DLB']['obsdate'].min()
            if is_dlb_within_one_year(pd_obsdate, dlb_obsdate):
                pd_list.append(patid)
            else:
                dlb_list.append(patid)
            continue    
        
        if has_pd and has_nondegen and not has_dlb and not has_degen:
            pd_list.append(patid)
            continue
        
        if has_nondegen and not has_pd and not has_dlb and not has_degen:
            filtered_records = patient_records[patient_records['Group'].isin(['VP', 'DIP', 'Secondary'])]
            recent_group = get_most_recent_group(filtered_records)
            #recent_group = get_most_recent_group(patient_records)
            if recent_group == 'VP':
                vp_list.append(patid)
            elif recent_group == 'DIP':
                dip_list.append(patid)
            elif recent_group == 'Secondary':
                sec_list.append(patid)
            continue
        
        if has_nondegen and has_dlb and not has_pd and not has_degen:
            dlb_list.append(patid)
            continue
        
        if has_nondegen and has_dlb and not has_pd and has_degen:
            filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS', 'DLB'])]
            recent_group = get_most_recent_group(filtered_records)
            if recent_group == 'MSA':
                msa_list.append(patid)
            elif recent_group == 'PSP':
                psp_list.append(patid)
            elif recent_group == 'CBS':
                cbs_list.append(patid)
            else:
                dlb_list.append(patid)
            continue    
        
        if has_nondegen and has_degen and not has_pd and not has_dlb:
            filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS'])]
            recent_group = get_most_recent_group(filtered_records)
            if recent_group == 'MSA':
                msa_list.append(patid)
            elif recent_group == 'PSP':
                psp_list.append(patid)
            elif recent_group == 'CBS':
                cbs_list.append(patid)
            continue    
        
        if has_nondegen and has_degen and has_pd and not has_dlb:
            filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS','PD'])]
            recent_group = get_most_recent_group(filtered_records)
            if recent_group == 'MSA':
                msa_list.append(patid)
            elif recent_group == 'PSP':
                psp_list.append(patid)
            elif recent_group == 'CBS':
                cbs_list.append(patid)
            elif recent_group == 'PD':
                pd_list.append(patid)    
            continue    
        
        if has_degen and not has_pd and has_dlb:
            filtered_records = patient_records[patient_records['Group'].isin(['MSA', 'PSP', 'CBS','DLB'])]
            recent_group = get_most_recent_group(filtered_records)
            if recent_group == 'MSA':
                msa_list.append(patid)
            elif recent_group == 'PSP':
                psp_list.append(patid)
            elif recent_group == 'CBS':
                cbs_list.append(patid) 
            elif recent_group == 'DLB':
                dlb_list.append(patid)    
        if not has_degen and has_pd and not has_dlb:
             pd_list.append(patid)

# Calculate the sum of all lists
total_patients = len(pd_list) + len(msa_list) + len(psp_list) + len(cbs_list) + len(dlb_list) + len(vp_list) + len(dip_list) + len(sec_list)

print("PD List:", len(pd_list))
print("MSA List:", len(msa_list))
print("PSP List:", len(psp_list))
print("CBS List:", len(cbs_list))
print("DLB List:", len(dlb_list))
print("VP List:", len(vp_list))
print("DIP List:", len(dip_list))
print("Secondary List:", len(sec_list))
print("Number of patients passing the first if condition:", patients_pass_first_condition)
print("Total number of patients:", total_patients)


# In[78]:


data = []

for patid in pd_list:
    data.append((patid, 'pd'))
for patid in msa_list:
    data.append((patid, 'msa'))
for patid in psp_list:
    data.append((patid, 'psp'))
for patid in cbs_list:
    data.append((patid, 'cbs'))
for patid in dlb_list:
    data.append((patid, 'dlb'))
for patid in vp_list:
    data.append((patid, 'vp'))
for patid in dip_list:
    data.append((patid, 'dip'))
for patid in sec_list:
    data.append((patid, 'secondary'))

# Convert the list of tuples into a DataFrame
df = pd.DataFrame(data, columns=['patid', 'Group'])

# Display the DataFrame
print(df)


# In[79]:


df['Group'].value_counts()


# In[ ]:


#Now this df would have list of intrested patienst and their diagnosis after the rukeset

