#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/llee/Downloads/4_nations_data_for_forestplot.csv')
df


# In[13]:


df = df.rename(columns={'Number of patients': 'Cases', 'Total patients': 'Denom'})


# In[14]:


df['Cases'] = df['Cases'].astype('float64')
df['Denom'] = df['Denom'].astype('float64')


# In[15]:


#Age encoding
age_band_mapping = {
    '0-19': 1, '20-24': 2, '25-29': 3, '30-34': 4, '35-39': 5, '40-44': 6, 
    '45-49': 7, '50-54': 8, '55-59': 9, '60-64': 10, '65-69': 11, 
    '70-74': 12, '75-79': 13, '80-84': 14, '85-89': 15, '90-94': 16, '95+': 17
}


df['age_band'] = df['age_band'].map(age_band_mapping)


# In[16]:


#Gender encoding

df = df[df.gender != 'Other']

df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

df.info()


# In[17]:


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


# In[18]:


#Nation encoding
#Sort so that England is the reference
nation_mapping = {
    'England': 'a_England',
    'Scotland': 'b_Scotland',
    'Wales': 'c_Wales',
    'Northern Ireland': 'd_Northern_Ireland',
}

df['nation'] = df['nation'].map(nation_mapping)

#Then create dummies
nation_dummies = pd.get_dummies(df['nation'], prefix = 'nation', drop_first = True)
df = pd.concat([df, nation_dummies], axis = 1)

df.info()


# In[19]:


urban_rural_mapping = {
    1: 0,
    2: 1,
}


df['urban_rural'] = df['urban_rural'].map(urban_rural_mapping)


# In[20]:


df['Prevalence_Rate'] = (df['Cases'] / df['Denom'])
df.info()


# In[21]:


df.isnull().any().any() 


# #Apply continuity correction
# df["Cases"] = np.where(df["Prevalence_Rate"] == 0, df["Cases"] + 0.1, df["Cases"])
# 
# df["Cases"] = np.where(df["Prevalence_Rate"] == 1, df["Cases"] - 0.1, df["Cases"])
# 
# df.max()

# In[22]:


#Build a Poisson model to predict the # cases using the log of the denominator as an offset
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','ethnicity_e_Unknown','nation_b_Scotland','nation_c_Wales','nation_d_Northern_Ireland']]
y = df['Cases']

X = sm.add_constant(X)

poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + ethnicity_e_Unknown + nation_b_Scotland + nation_c_Wales + nation_d_Northern_Ireland",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[23]:


#Residual diagnostics

plt.scatter(poisson_result.fittedvalues, poisson_result.resid_pearson)
plt.axhline(0, color='red', linestyle = '--')
plt.xlabel('Fitted values')
plt.ylabel('Pearson residuals')
plt.title('Residuals vs Fitted Values (Poisson)')
plt.show()


# In[24]:


#Check if residuals are symmetrical

plt.hist(poisson_result.resid_pearson, bins= 30, edgecolor = 'black')
plt.title('Histogram of Pearson residuals (Poisson)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[25]:


fig = sm.qqplot(poisson_result.resid_deviance, line='45', fit=True)
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()


# In[26]:


plt.scatter(poisson_result.fittedvalues, df['Cases'])
plt.plot(y, y, '--', label='y = x')
plt.xlabel('Fitted values')
plt.ylabel('Actual cases')
plt.title('Fitted Values vs Actuals (Poisson)')
plt.show()


# In[27]:


#Check for overdispersion
residual_deviance = poisson_result.deviance
residual_df = poisson_result.df_resid
dispersion_param = residual_deviance / residual_df

print("Deviance-based dispersion parameter (Ordinal model):", dispersion_param)


# In[28]:


odds_ratios = np.exp(poisson_result.params)
print(odds_ratios)


# In[29]:


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


# In[30]:


print(poisson_result.fittedvalues)


# df['poisson_fitted'] = poisson_result.fittedvalues
# 
# df.to_csv('G:\My Drive\RED_pois.csv')

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
    'nation_b_Scotland': 'Scotland',
    'nation_c_Wales': 'Wales',
    'nation_d_Northern_Ireland': 'Northern Ireland',
    'age_band': 'Per 5 years'
}


odds_ratios_df['label'] = odds_ratios_df['label'].map(label_mapping)

group_mapping = {
    'Male': 'Gender (compared to Female)',
    'Asian': 'Ethnicity (compared to White)',
    'African or Caribbean': 'Ethnicity (compared to White)',
    'Mixed or Other': 'Ethnicity (compared to White)',
    'Ethnicity unknown': 'Ethnicity (compared to White)',
    'Rural': 'Dwelling (compared to Urban)',
    'Scotland': 'Nation (compared to England)',
    'Wales': 'Nation (compared to England)',
    'Northern Ireland': 'Nation (compared to England)',
    'Per 5 years': 'Age band'
}

odds_ratios_df['group'] = odds_ratios_df['label'].map(group_mapping)

odds_ratios_df.info()

odds_ratios_df


# In[32]:


forest = fp.forestplot(odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
              ylabel = "Odds ratio (95% Confidence interval)", 
              xlabel = "Odds ratio",
              xticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Gender (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Nation (compared to England)"],
              sort = False,
              **{
                  "marker": "D"
              }
             )

forest.axvline(1, linewidth = 1, color = '#808080')

plt.savefig("C:/Users/llee/Downloads/RED odds ratios_prevalence23_with_nation_20250120.png", dpi=300, bbox_inches="tight")

