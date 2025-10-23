#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')
df


# In[2]:


df['uksp_rate'] = df['pd'] / df['denom'] * 100000


df.info()


# # a. Pre-Post Covid, PD only

# In[3]:


#

pdf=df[['year','age','gender','denom','uksp_rate']]

pdf=pdf[pdf['year'] < 2020]

pdf=pdf[pdf['year'] > 2015]

pdf


# In[4]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender'])[["uksp_rate"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['gender'])
    .assign(intercept = 1)
    .sort_values(['year','age','gender_female'])
    .reset_index(drop=True)
    )

y = X.pop("uksp_rate")


# In[5]:


X['age_sq'] = X['age']**2
X


# In[6]:


y


# In[7]:


#1st model: intercept only with no indicator variables

import statsmodels.api as sm

model_no_indicators = sm.GLM(y, X["intercept"],
                            family=sm.families.Poisson(),
                            )
result_no_indicators = model_no_indicators.fit()
print(result_no_indicators.summary())


# In[8]:


#Plot fitted values against observed values 

import matplotlib.pyplot as plt

plt.plot(y, result_no_indicators.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[9]:


#2nd model: add in age and gender

model_age_gender = sm.GLM(y, X[["intercept","age","gender_male"]],
                            family=sm.families.Poisson(),
                            )
result_age_gender = model_age_gender.fit()
print(result_age_gender.summary())


# In[10]:


plt.plot(y, result_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[11]:


#3rd model: add in year alongside age and gender

model_year_age_gender = sm.GLM(y, X[["intercept","year","age","gender_male"]],
                            family=sm.families.Poisson(),
                            )
result_year_age_gender = model_year_age_gender.fit()
print(result_year_age_gender.summary())


# In[12]:


plt.plot(y, result_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[13]:


#4th model: add in squared age to reflect polynomial behaviour of the input variable

model_year_age_gender = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                            family=sm.families.Poisson(),
                            )
result_year_age_gender = model_year_age_gender.fit()
print(result_year_age_gender.summary())


# In[14]:


plt.plot(y, result_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[15]:


#Converting model parameters to odds ratios

coefficients = result_year_age_gender.params
conf_int = result_year_age_gender.conf_int()
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


# In[29]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result_year_age_gender.fittedvalues
X['deviance_resids'] = result_year_age_gender.resid_deviance
X['pd'] = y
X.to_csv('G:\My Drive\poisson_data.csv')
X


# In[16]:


#TEST OF EQUIDISPERSION

# Get model deviance residuals
deviance_residuals = result_year_age_gender.resid_deviance

# Calculate residual deviance
residual_deviance = sum(deviance_residuals)

# Calculate degrees of freedom
df_res = result_year_age_gender.df_resid

residual_deviance

# Calculate the ratio
ratio = residual_deviance / df_res

# Display the ratio
print("Residual Deviance:", residual_deviance)
print("Degrees of Freedom:", df_res)
print("Residual Deviance to Degrees of Freedom Ratio:", ratio)


# In[21]:


#Obtain the value of alpha for the negative binomial model

import statsmodels.formula.api as smf


pdf['LAMBDA'] = result_year_age_gender.mu

'''
Then we add a new column to our dataframe, 
which is derived from the λ vector.
It will serve as the dependent variable for our auxiliary OLS regression
'''

pdf['AUX_OLS'] = pdf.apply(lambda x: ((x['uksp_rate'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)

# Specify the aux. OLS model 
ols_expr = """AUX_OLS ~ LAMBDA - 1"""

# Fit the aux. OLS model based on above 
# expression and our dataframe insurance_data
aux_olsr_results = smf.ols(ols_expr, pdf).fit()

# Print regression parameters
print(aux_olsr_results.params)


# In[22]:


#Is alpha statistically significant?
aux_olsr_results.tvalues

#0.007 < t-value at 95% confidence (right-tailed) with 131d.f. = 1.66; ergo alpha is not statistically significant


# In[17]:


#Ask the model to project forwards
#Create a test dataframe X_test with years 2020-2050
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test



# In[18]:


X_test = test[['year','age','gender_female','gender_male','denom','intercept']]

X_test['age_sq'] = X_test['age']**2

X_test=X_test[X_test['year'] < 2024]

X_test



# In[19]:


pois_predictions = result_year_age_gender.get_prediction(X_test[["intercept","year","age","age_sq","gender_male"]], 
                                                        )

pois_predictions_summary = pois_predictions.summary_frame()
print(pois_predictions_summary)


# In[20]:


#Add the predictions back into the test dataframe 

X_test['UKSP_rate_pred'] = pois_predictions_summary['mean']
X_test['UKSP_rate_lower'] = pois_predictions_summary['mean_ci_lower']
X_test['UKSP_rate_upper'] = pois_predictions_summary['mean_ci_upper']

X_test


# In[21]:


#Calculate absolute no.s of PD cases in the UK

X_test['Abs_case_midpoint'] = X_test['denom'] * X_test['UKSP_rate_pred'] / 100000
X_test['Abs_case_lower'] = X_test['denom'] * X_test['UKSP_rate_lower'] / 100000
X_test['Abs_case_upper'] = X_test['denom'] * X_test['UKSP_rate_upper'] / 100000

X_test


# In[22]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250731_PD_Inc_Pre-PostCovid.csv')


# In[27]:


#Actuals
adf=df[['year','pd']]

adf=adf[adf['year'] >= 2020]

adf['pd'].sum()

