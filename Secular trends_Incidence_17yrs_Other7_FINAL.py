#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !/usr/bin/env python
# coding: utf-8

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')

df


# In[2]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[3]:


print(df.groupby('gender')['other_7'].sum().reset_index(), " ")


# In[4]:


df.info()


#   # b. 17 years 2003-19, Other 7 Parkinsonisms

# In[5]:


#Aggregate to annual totals
df_agg = (df
    .groupby(['year'])[["other_7"]]
    .sum()
    .reset_index()
    .reset_index(drop=True)
    )
df_agg


# In[6]:


#

pdf=df[['year','age','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2020]

pdf=pdf[pdf['year'] > 2012] 

pdf


# In[7]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age_band','gender','ln_denom'])[["other_7"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age_band','gender'])
    .assign(intercept = 1)
    .sort_values(['year',"gender_female","age_band_0-19","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+"])
    .reset_index(drop=True)
    )

y = X.pop("other_7")

X


# In[14]:


#X['age_sq'] = X['age']**2
#X['age_cubed'] = X['age']**3
#X['age_ln'] = np.log(X['age'])
#X['age_sqrt'] = np.sqrt(X['age'])
#X['year_sq'] = X['year']**2
#X['year_cubed'] = X['year']**3
#X['year_ln'] = np.log(X['year'])
#X['year_sqrt'] = np.sqrt(X['year'])

X


# In[15]:


y


# In[18]:


#1st model: intercept only with no indicator variables

import statsmodels.api as sm

model_no_indicators = sm.GLM(y, X["intercept"],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_no_indicators = model_no_indicators.fit()
print(result_no_indicators.summary())


# In[19]:


#Plot fitted values against observed values 

import matplotlib.pyplot as plt

plt.plot(y, result_no_indicators.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[20]:


#2nd model: add in age, gender and year

model_age_year = sm.GLM(y, X[["intercept","year","gender_male",'age_22.5','age_27.5','age_32.5','age_37.5','age_42.5','age_47.5','age_52.5','age_57.5','age_62.5','age_67.5','age_72.5','age_77.5','age_82.5','age_87.5','age_92.5','age_97.5']],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_age_year = model_age_year.fit()
print(result_age_year.summary())


# In[21]:


plt.plot(y, result_age_year.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[23]:


#Plot fitted values against residuals

residuals = y - result_age_year.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result_age_year.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[24]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[25]:


#Converting model parameters to odds ratios

coefficients = result_age_year.params
conf_int = result_age_year.conf_int()
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


# In[ ]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result_age_year.fittedvalues
X['deviance_resids'] = result_age_year.resid_deviance
X['pd'] = y
#X.to_csv('G:\My Drive\poisson_data.csv')


# In[26]:


#See what happens when we ask the model to project forwards
#Create a test dataframe X_test with years 2019-2023
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test


# In[27]:


X_test = test[['year','age','gender_female','gender_male','denom','log_denom','intercept']]

#X_test['age_sq'] = X_test['age']**2

X_test['log_denom'] = np.log(X_test['denom'])

X_test



# In[28]:


X_test = (X_test
    .pipe(pd.get_dummies, columns=['age'])
    )

X_test


# In[34]:


model_predictions = result_age_year.get_prediction(X_test[["intercept","year","gender_male",'age_22.5','age_27.5','age_32.5','age_37.5','age_42.5','age_47.5','age_52.5','age_57.5','age_62.5','age_67.5','age_72.5','age_77.5','age_82.5','age_87.5','age_92.5','age_97.5']], 
                                                   offset=X_test["log_denom"],
                                                  )

model_predictions_summary = model_predictions.summary_frame()
print(model_predictions_summary)


# In[35]:


#Add the predictions back into the test dataframe 

X_test['case_cnt_pred'] = model_predictions_summary['mean']
X_test['case_cnt_lower'] = model_predictions_summary['mean_ci_lower']
X_test['case_cnt_upper'] = model_predictions_summary['mean_ci_upper']


# In[36]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250402_Other7_Inc_Predictions.csv')


# In[37]:


#Aggregate to annual totals
X_agg = (X_test
    .groupby(['year'])[["case_cnt_pred","case_cnt_lower",'case_cnt_upper']]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
X_agg


# # OLS model built on 2011-19 for future projections

# In[1]:


#FINAL MODEL: Use simple linear regression from the 2011-19 data

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')
df


# In[2]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[3]:


#

pdf=df[['year','age','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2020]

pdf=pdf[pdf['year'] > 2010] 

pdf


# In[4]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender','ln_denom'])[["other_7"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age','gender'])
    .assign(intercept = 1)
    .sort_values(['year',"gender_female","age_10.0","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5"])
    .reset_index(drop=True)
    )

y = X.pop("other_7")

X


# In[5]:


#Build the OLS model

import statsmodels.api as sm

model = sm.OLS(y, X[["intercept","year","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5","gender_male"]])

result = model.fit()
print(result.summary())


# In[6]:


import matplotlib.pyplot as plt

plt.plot(y, result.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[7]:


#Plot fitted values against residuals

residuals = y - result.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[8]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[9]:


coefficients = result.params
conf_int = result.conf_int()
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


# In[10]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result.fittedvalues
X['pd'] = y
X.to_csv('G:\My Drive\Inc_Other_7_fitted_FINAL.csv')


# In[17]:


#See what happens when we ask the model to project forwards
#Create a test dataframe X_test with years 2019-2023
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test


# In[18]:


X_test = test[['year','age','gender_female','gender_male','denom','log_denom','intercept']]

#X_test['age_sq'] = X_test['age']**2

X_test['log_denom'] = np.log(X_test['denom'])

X_test


# In[19]:


X_test = (X_test
    .pipe(pd.get_dummies, columns=['age'])
    )

X_test


# In[20]:


model_predictions = result.get_prediction(X_test[["intercept","year","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5","gender_male"]], 
                                                      )

model_predictions_summary = model_predictions.summary_frame()
print(model_predictions_summary)


# In[21]:


#Add the predictions back into the test dataframe 

X_test['case_cnt_pred'] = model_predictions_summary['mean']
X_test['case_cnt_lower'] = model_predictions_summary['mean_ci_lower']
X_test['case_cnt_upper'] = model_predictions_summary['mean_ci_upper']


# In[22]:


#Export to csv
X_test.to_csv('G:\My Drive\\2020728_Other7_Inc_Predictions_v4.csv')


# # OLS model built on 2003-13

# In[11]:


#FINAL MODEL: Use simple linear regression from the 2003-13 data

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')
df


# In[12]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[13]:


#

pdf=df[['year','age','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2014]

pdf


# In[14]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender','ln_denom'])[["other_7"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age','gender'])
    .assign(intercept = 1)
    .sort_values(['year',"gender_female","age_10.0","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5"])
    .reset_index(drop=True)
    )

y = X.pop("other_7")

X


# In[15]:


#Build the OLS model

import statsmodels.api as sm

model = sm.OLS(y, X[["intercept","year","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5","gender_male"]])

result = model.fit()
print(result.summary())


# In[28]:


import matplotlib.pyplot as plt

plt.plot(y, result.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[29]:


#Plot fitted values against residuals

residuals = y - result.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[30]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[30]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result.fittedvalues
X['other_7'] = y
X.to_csv('G:\My Drive\Inc_Other_7_fitted_Early.csv')


# In[31]:


new_data=pd.read_csv('C:/Users/llee/Downloads/Gradient_incidence - Sheet1.csv')

new_data = new_data[["intercept","year","age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5","gender_male"]]
new_data


# In[32]:


new_data_2003 = new_data[new_data['year'] == 2003]
new_data_2010 = new_data[new_data['year'] == 2010]
new_data_2003


# In[34]:


predictions_2003 = result.get_prediction(new_data_2003)
pred_int_2003 = predictions_2003.summary_frame(alpha=0.05)
predicted_value_2003 = pred_int_2003['mean'][0]
lower_bound_2003 = pred_int_2003['obs_ci_lower'][0]
upper_bound_2003 = pred_int_2003['obs_ci_upper'][0]

print(f"PRediction for 2003: {predicted_value_2003:.4f}")
print(f"95% Prediction interval for 2003: [{lower_bound_2003:.4f}, {upper_bound_2003:.4f}]")


# In[35]:


result.model.exog_names

