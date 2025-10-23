#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/llee/Downloads/Incidence_2023_firsthalf_data_for_forestplot.csv')
df


# In[2]:


df = df.rename(columns={'Number of patients': 'Cases', 'Total patients': 'Denom'})


# In[3]:


df['Cases'] = df['Cases'].astype('float64')
df['Denom'] = df['Denom'].astype('float64')


# In[4]:


#Age encoding
age_band_mapping = {
    '0-19': 1, '20-24': 2, '25-29': 3, '30-34': 4, '35-39': 5, '40-44': 6, 
    '45-49': 7, '50-54': 8, '55-59': 9, '60-64': 10, '65-69': 11, 
    '70-74': 12, '75-79': 13, '80-84': 14, '85-89': 15, '90-94': 16, '95+': 17
}


df['age_band'] = df['age_band'].map(age_band_mapping)


# In[5]:


#Gender encoding

df = df[df.gender != 'Other']

df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

df.info()


# In[6]:


#Ethnicity encoding
#Sort so that White is the reference
ethnicity_mapping = {
    'White': 'a_White',
    'Asian': 'b_Asian',
    'Mixed or Other': 'd_Mixed_or_Other',
    'African or Caribbean': 'c_African_or_Caribbean',
    'Unknown': 'e_Unknown'
}

df['ethnicity'] = df['ethnicity'].map(ethnicity_mapping)

#Then create dummies
ethnicity_dummies = pd.get_dummies(df['ethnicity'], prefix = 'ethnicity', drop_first = True)
df = pd.concat([df, ethnicity_dummies], axis = 1)

df.info()


# In[7]:


#IMD: Create two versions, ordinal and nominal 

imd_mapping = {
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    'Unknown': 6
}


df['imd_ordinal'] = df['imd'].map(imd_mapping)

imd_dummies = pd.get_dummies(df['imd'], prefix = 'imd', drop_first = True)
df = pd.concat([df, imd_dummies], axis = 1)

df.info()


# In[8]:


urban_rural_mapping = {
    1: 0,
    2: 1,
}


df['urban_rural'] = df['urban_rural'].map(urban_rural_mapping)


# In[9]:


df['Prevalence_Rate'] = (df['Cases'] / df['Denom'])
df.info()


# In[10]:


df.isnull().any().any() 


# In[11]:


#Build a Poisson model to predict the # cases using the log of the denominator as an offset
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','ethnicity_e_Unknown','imd_2','imd_3','imd_4','imd_5','imd_Unknown']]
y = df['Cases']

X = sm.add_constant(X)

#poisson_result = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=df['Denom']).fit()
poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + ethnicity_e_Unknown + imd_2 + imd_3 + imd_4 + imd_5 + imd_Unknown",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[21]:


#Residual diagnostics

plt.scatter(poisson_result.fittedvalues, poisson_result.resid_pearson)
plt.axhline(0, color='red', linestyle = '--')
plt.xlabel('Fitted values')
plt.ylabel('Pearson residuals')
plt.title('Residuals vs Fitted Values (Poisson)')
plt.show()


# In[22]:


#Check if residuals are symmetrical

plt.hist(poisson_result.resid_pearson, bins= 30, edgecolor = 'black')
plt.title('Histogram of Pearson residuals (Poisson)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[23]:


fig = sm.qqplot(poisson_result.resid_deviance, line='45', fit=True)
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()


# In[24]:


plt.scatter(poisson_result.fittedvalues, df['Cases'])
plt.plot(y, y, '--', label='y = x')
plt.xlabel('Fitted values')
plt.ylabel('Actual cases')
plt.title('Fitted Values vs Actuals (Poisson)')
plt.show()


# In[12]:


#Check for overdispersion

def check_overdispersion(model):
    resid_dev = model.deviance
    df_resid = model.df_resid
    dispersion = resid_dev / df_resid
    print("Dispersion:", dispersion)
    return dispersion

check_overdispersion(poisson_result)


# In[26]:


odds_ratios = np.exp(poisson_result.params)
print(odds_ratios)


# In[30]:


coefficients = poisson_result.params
conf_int = poisson_result.conf_int()
conf_int.columns = ['2.5%', '97.5%']  # Renaming columns for clarity
odds_ratios = np.exp(coefficients)
conf_int_odds_ratios = np.exp(conf_int)

# Displaying coefficients and their 95% CI
print("\nCoefficients and 95% CI:")
coefficients_df = pd.DataFrame({
    "Coefficient": coefficients,
    "2.5% CI (Coeff)": conf_int["2.5%"],
    "97.5% CI (Coeff)": conf_int["97.5%"]
})
print(coefficients_df)

# Displaying Odds Ratios and their 95% CI
print("\nOdds Ratios and 95% CI:")
odds_ratios_df = pd.DataFrame({
    "Odds Ratio": odds_ratios,
    "2.5% CI (OR)": conf_int_odds_ratios["2.5%"],
    "97.5% CI (OR)": conf_int_odds_ratios["97.5%"]
})
print(odds_ratios_df)


# In[63]:


print(poisson_result.fittedvalues)


# In[23]:


df['poisson_fitted'] = poisson_result.fittedvalues

df.to_csv('G:\My Drive\RED_pois.csv')


# In[31]:


import forestplot as fp

odds_ratios_df = odds_ratios_df.rename_axis("label").reset_index()

odds_ratios_df = odds_ratios_df[odds_ratios_df.label != 'Intercept']

label_mapping = {
    'gender': 'Male',
    'ethnicity_b_Asian': 'Asian',
    'ethnicity_c_African_or_Caribbean': 'African or Caribbean',
    'ethnicity_d_Mixed_or_Other': 'Mixed or Other',
    'ethnicity_e_Unknown': 'Ethnicity unknown',
    'urban_rural': 'Rural',
    'imd_2': 'IMD 4',
    'imd_3': 'IMD 3',
    'imd_4': 'IMD 2',
    'imd_5': 'IMD 1 (most deprived)',
    'imd_Unknown': 'IMD unknown',
    'age_band': 'Per 5 years'
}


odds_ratios_df['label'] = odds_ratios_df['label'].map(label_mapping)

group_mapping = {
    'Male': 'Sex (compared to Female)',
    'Asian': 'Ethnicity (compared to White)',
    'African or Caribbean': 'Ethnicity (compared to White)',
    'Mixed or Other': 'Ethnicity (compared to White)',
    'Ethnicity unknown': 'Ethnicity (compared to White)',
    'Rural': 'Dwelling (compared to Urban)',
    'IMD 4': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 3': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 2': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 1 (most deprived)': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD unknown': 'Deprivation (compared to IMD 5 least deprived)',
    'Per 5 years': 'Age band'
}

odds_ratios_df['group'] = odds_ratios_df['label'].map(group_mapping)

odds_ratios_df.info()

odds_ratios_df


# In[32]:


#Export to .csv to enable it to be combined with the prevalence chart in a single image (refer to prevalence code)

odds_ratios_df.to_csv('G:\My Drive\\20250404_RED_Incidence_chartdata.csv')


# In[69]:


forest = fp.forestplot(odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
              ylabel = "Odds ratio (95% Confidence interval)", 
              xlabel = "Odds ratio",
              xticks=[0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Gender (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)"],
              sort = False,
              **{
                  "marker": "D"
              }
             )

forest.axvline(1, linewidth = 1, color = '#808080')

plt.savefig("C:/Users/llee/Downloads/RED odds ratios_incidence23_H1_20250124.png", dpi=300, bbox_inches="tight")


# In[23]:


###########################################################
#Too many missing ethnicity records - try excluding them###
###########################################################

df=pd.read_csv('C:/Users/llee/Downloads/Incidence_2023_data_for_forestplot.csv')

df = df.rename(columns={'Number of patients': 'Cases', 'Total patients': 'Denom'})

df['Cases'] = df['Cases'].astype('float64')
df['Denom'] = df['Denom'].astype('float64')

#Age encoding
age_band_mapping = {
    '0-19': 1, '20-24': 2, '25-29': 3, '30-34': 4, '35-39': 5, '40-44': 6, 
    '45-49': 7, '50-54': 8, '55-59': 9, '60-64': 10, '65-69': 11, 
    '70-74': 12, '75-79': 13, '80-84': 14, '85-89': 15, '90-94': 16, '95+': 17
}


df['age_band'] = df['age_band'].map(age_band_mapping)

#Gender encoding

df = df[df.gender != 'Other']

df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

df.info()


# In[24]:


#Ethnicity encoding

df = df[df.ethnicity != 'Unknown']

ethnicity_mapping = {
    'White': 'a_White',
    'Asian': 'b_Asian',
    'Mixed or Other': 'd_Mixed_or_Other',
    'African or Caribbean': 'c_African_or_Caribbean',
}

df['ethnicity'] = df['ethnicity'].map(ethnicity_mapping)

#Then create dummies
ethnicity_dummies = pd.get_dummies(df['ethnicity'], prefix = 'ethnicity', drop_first = True)
df = pd.concat([df, ethnicity_dummies], axis = 1)

df.info()


# In[25]:


#IMD: Create two versions, ordinal and nominal 

imd_mapping = {
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    'Unknown': 6
}


df['imd_ordinal'] = df['imd'].map(imd_mapping)

imd_dummies = pd.get_dummies(df['imd'], prefix = 'imd', drop_first = True)
df = pd.concat([df, imd_dummies], axis = 1)

#Urban-Rural
urban_rural_mapping = {
    1: 0,
    2: 1,
}


df['urban_rural'] = df['urban_rural'].map(urban_rural_mapping)

df['Prevalence_Rate'] = (df['Cases'] / df['Denom'])
df.info()


# In[26]:


#Build a Poisson model to predict the # cases instead using the log of the denominator as an offset
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','imd_2','imd_3','imd_4','imd_5','imd_Unknown']]
y = df['Cases']

X = sm.add_constant(X)

#poisson_result = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=df['Denom']).fit()
poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + imd_2 + imd_3 + imd_4 + imd_5 + imd_Unknown",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[27]:


#Residual diagnostics

plt.scatter(poisson_result.fittedvalues, poisson_result.resid_pearson)
plt.axhline(0, color='red', linestyle = '--')
plt.xlabel('Fitted values')
plt.ylabel('Pearson residuals')
plt.title('Residuals vs Fitted Values (Poisson)')
plt.show()


# In[28]:


#Check if residuals are symmetrical

plt.hist(poisson_result.resid_pearson, bins= 30, edgecolor = 'black')
plt.title('Histogram of Pearson residuals (Poisson)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[29]:


fig = sm.qqplot(poisson_result.resid_deviance, line='45', fit=True)
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()


# In[30]:


plt.scatter(poisson_result.fittedvalues, df['Cases'])
plt.plot(y, y, '--', label='y = x')
plt.xlabel('Fitted values')
plt.ylabel('Actual cases')
plt.title('Fitted Values vs Actuals (Poisson)')
plt.show()


# In[31]:


#Check for overdispersion
residual_deviance = poisson_result.deviance
residual_df = poisson_result.df_resid
dispersion_param = residual_deviance / residual_df

print("Deviance-based dispersion parameter (Ordinal model):", dispersion_param)


# In[32]:


odds_ratios = np.exp(poisson_result.params)
print(odds_ratios)


# In[33]:


coefficients = poisson_result.params
conf_int = poisson_result.conf_int()
conf_int.columns = ['2.5%', '97.5%']  # Renaming columns for clarity
odds_ratios = np.exp(coefficients)
conf_int_odds_ratios = np.exp(conf_int)

# Displaying coefficients and their 95% CI
print("\nCoefficients and 95% CI:")
coefficients_df = pd.DataFrame({
    "Coefficient": coefficients,
    "2.5% CI (Coeff)": conf_int["2.5%"],
    "97.5% CI (Coeff)": conf_int["97.5%"]
})
print(coefficients_df)

# Displaying Odds Ratios and their 95% CI
print("\nOdds Ratios and 95% CI:")
odds_ratios_df = pd.DataFrame({
    "Odds Ratio": odds_ratios,
    "2.5% CI (OR)": conf_int_odds_ratios["2.5%"],
    "97.5% CI (OR)": conf_int_odds_ratios["97.5%"]
})
print(odds_ratios_df)


# In[34]:


import forestplot as fp

odds_ratios_df = odds_ratios_df.rename_axis("label").reset_index()

odds_ratios_df = odds_ratios_df[odds_ratios_df.label != 'Intercept']

label_mapping = {
    'gender': 'Male',
    'ethnicity_b_Asian': 'Asian',
    'ethnicity_c_African_or_Caribbean': 'African or Caribbean',
    'ethnicity_d_Mixed_or_Other': 'Mixed or Other',
    'urban_rural': 'Rural',
    'imd_2': 'IMD 4',
    'imd_3': 'IMD 3',
    'imd_4': 'IMD 2',
    'imd_5': 'IMD 1 (most deprived)',
    'imd_Unknown': 'IMD unknown',
    'age_band': 'Per 5 years'
}


odds_ratios_df['label'] = odds_ratios_df['label'].map(label_mapping)

group_mapping = {
    'Male': 'Gender (compared to Female)',
    'Asian': 'Ethnicity (compared to White)',
    'African or Caribbean': 'Ethnicity (compared to White)',
    'Mixed or Other': 'Ethnicity (compared to White)',
    'Rural': 'Dwelling (compared to Urban)',
    'IMD 4': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 3': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 2': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD 1 (most deprived)': 'Deprivation (compared to IMD 5 least deprived)',
    'IMD unknown': 'Deprivation (compared to IMD 5 least deprived)',
    'Per 5 years': 'Age band'
}

odds_ratios_df['group'] = odds_ratios_df['label'].map(group_mapping)

odds_ratios_df.info()

odds_ratios_df


# In[35]:


forest = fp.forestplot(odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
              ylabel = "Odds ratio (95% Confidence interval)", 
              xlabel = "Odds ratio",
              xticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Gender (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)"],
              sort = False,
              **{
                  "marker": "D"
              }
             )

forest.axvline(1, linewidth = 1, color = '#808080')

#plt.savefig("C:/Users/llee/Downloads/RED odds ratios_prevalence23_20250114.png", dpi=300, bbox_inches="tight")

