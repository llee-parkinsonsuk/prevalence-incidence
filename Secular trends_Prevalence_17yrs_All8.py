#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !/usr/bin/env python
# coding: utf-8

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V1_20250313_Secular_trends_PREV_all_years.csv')
df


# In[2]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['all_8'] = df['pd']+df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[3]:


print(df.groupby('gender')['all_8'].sum().reset_index(), " ",
      df.groupby('gender')['denom'].sum().reset_index(), " ",
      df.groupby('gender')['ln_denom'].sum().reset_index())


# In[4]:


df.info()


#   # b. 17 years 2003-19, All 8 Parkinsonisms (PD + Others)

# In[5]:


#

pdf=df[['year','age','gender','ln_denom','denom','all_8']]

pdf=pdf[pdf['year'] < 2020]

pdf


# In[7]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender'])[["all_8","ln_denom"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['gender'])
    .assign(intercept = 1)
    .sort_values(['year','age','gender_female'])
    .reset_index(drop=True)
    )

y = X.pop("all_8")


# In[8]:


X['age_sq'] = X['age']**2
X


# In[9]:


y


# In[10]:


#1st model: intercept only with no indicator variables

import statsmodels.api as sm

model_no_indicators = sm.GLM(y, X["intercept"],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_no_indicators = model_no_indicators.fit()
print(result_no_indicators.summary())


# In[11]:


#Plot fitted values against observed values 

import matplotlib.pyplot as plt

plt.plot(y, result_no_indicators.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[12]:


#2nd model: add in age and gender

model_age_gender = sm.GLM(y, X[["intercept","age","gender_male"]],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_age_gender = model_age_gender.fit()
print(result_age_gender.summary())


# In[13]:


plt.plot(y, result_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[14]:


#3rd model: add in year alongside age and gender

model_year_age_gender = sm.GLM(y, X[["intercept","year","age","gender_male"]],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_year_age_gender = model_year_age_gender.fit()
print(result_year_age_gender.summary())


# In[15]:


plt.plot(y, result_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[16]:


#4th model: add in squared age to reflect polynomial behaviour of the input variable

model_year_age_gender = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_year_age_gender = model_year_age_gender.fit()
print(result_year_age_gender.summary())


# In[17]:


plt.plot(y, result_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[18]:


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


# In[19]:


#Odds ratio for year is 0.995022; annual decrease therefore
annual_chg = np.exp(coefficients['year']) - 1
print("Annual change in prevalence = ", annual_chg*100, "%")

period_chg = annual_chg * 17
print("Total change in prevalence over period = ", period_chg*100, "%")


# In[20]:


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


# In[40]:


#Export actuals and model values to CSV for visualisation

#X['fittedvalues'] = result_year_age_gender.fittedvalues
#X['deviance_resids'] = result_year_age_gender.resid_deviance
#X['pd'] = y
#X.to_csv('G:\My Drive\poisson_data.csv')


# In[21]:


#Obtain the value of alpha for the negative binomial model

import statsmodels.formula.api as smf


pdf['LAMBDA'] = result_year_age_gender.mu

'''
Then we add a new column to our dataframe, 
which is derived from the λ vector.
It will serve as the dependent variable for our auxiliary OLS regression
'''

pdf['AUX_OLS'] = pdf.apply(lambda x: ((x['all_8'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)

# Specify the aux. OLS model 
ols_expr = """AUX_OLS ~ LAMBDA - 1"""

# Fit the aux. OLS model
aux_olsr_results = smf.ols(ols_expr, pdf).fit()

# Print regression parameters
print(aux_olsr_results.params)


# In[22]:


#Is alpha statistically significant?
aux_olsr_results.tvalues


# In[23]:


#Build the negative binomial model using the value of alpha we have obtained

nb_model_year_age_gender = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                                  offset=X["ln_denom"],
                                  family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0]))
result_nb_year_age_gender = nb_model_year_age_gender.fit()

print(result_nb_year_age_gender.summary())


# In[25]:


#Deviance
print ("Deviance:", result_nb_year_age_gender.deviance)
print ("Pearson's chi-sq:", result_nb_year_age_gender.pearson_chi2)

#critical value of chisq at 1% significance with 578d.f. ~= approx. 656
#Deviance and Pearson's chi-sq both < 656 ~ NB model provides good fit overall


# In[26]:


plt.plot(y, result_nb_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[27]:


#Converting model parameters to odds ratios

coefficients = result_nb_year_age_gender.params
conf_int = result_nb_year_age_gender.conf_int()
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


# In[28]:


#Odds ratio for year is 1.007227; annual INcrease therefore
annual_chg = np.exp(coefficients['year']) - 1
print("Annual change in prevalence = ", annual_chg*100, "%")

period_chg = annual_chg * 17
print("Total change in prevalence over period = ", period_chg*100, "%")


# In[38]:


#Export actuals and model values to CSV for visualisation

#X['nb_fittedvalues'] = result_nb_year_age_gender.fittedvalues
#X['nb_deviance_resids'] = result_nb_year_age_gender.resid_deviance
#X.to_csv('G:\My Drive\poisson_data.csv')


# In[29]:


#Ask the two models to project forwards
#Create a test dataframe X_test with years 2019-2023
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test



# In[30]:


X_test = test[['year','age','gender_female','gender_male','denom','log_denom','intercept']]

X_test['age_sq'] = X_test['age']**2

X_test['log_denom'] = np.log(X_test['denom'])

X_test



# In[31]:


pois_predictions = result_year_age_gender.get_prediction(X_test[["intercept","year","age","age_sq","gender_male"]], 
                                  offset=X_test["log_denom"])
nb2_predictions = result_nb_year_age_gender.get_prediction(X_test[["intercept","year","age","age_sq","gender_male"]], 
                                  offset=X_test["log_denom"])


pois_predictions_summary = pois_predictions.summary_frame()
print(pois_predictions_summary)

nb2_predictions_summary = nb2_predictions.summary_frame()
print(nb2_predictions_summary)


# In[32]:


#Add the predictions back into the test dataframe 

X_test['Poisson_pred'] = pois_predictions_summary['mean']
X_test['Poisson_lower'] = pois_predictions_summary['mean_ci_lower']
X_test['Poisson_upper'] = pois_predictions_summary['mean_ci_upper']
X_test['NB_pred'] = nb2_predictions_summary['mean']
X_test['NB_lower'] = nb2_predictions_summary['mean_ci_lower']
X_test['NB_upper'] = nb2_predictions_summary['mean_ci_upper']

X_test


# In[33]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250316_All8_Prev_2003-19.csv')


# In[34]:


#Aggregate to annual totals
X_agg = (X_test
    .groupby(['year'])[["Poisson_pred","Poisson_lower",'Poisson_upper','NB_pred',"NB_lower","NB_upper"]]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
X_agg


# In[35]:


#To create a comparison plot

predicted_poisson=X_agg['Poisson_pred']
predicted_nb = X_agg['NB_pred']
fig = plt.figure()
fig.suptitle('Predicted pwp counts')
pois_predicted, = plt.plot(X_agg.index, predicted_poisson, 'go-', label='Poisson predicted counts')
nb2_predicted, = plt.plot(X_agg.index, predicted_nb, 'ro-', label='Negative binomial predicted counts')
plt.legend(handles=[pois_predicted, nb2_predicted])
plt.show()


# In[ ]:




