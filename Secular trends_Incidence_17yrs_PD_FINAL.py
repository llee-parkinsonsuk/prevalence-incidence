#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')

df['uksp_rate'] = (df['pd'] / df['denom']) * 100000

df


# In[2]:


df.info()


# # a. 2003-19, PD only

# In[3]:


#

pdf=df[['year','age','gender','denom','uksp_rate']]

pdf=pdf[pdf['year'] < 2020]

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


# In[16]:


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


# In[17]:


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


# In[18]:


#Is alpha statistically significant?
aux_olsr_results.tvalues

#0.014 < t-value at 95% confidence (right-tailed) with 578d.f. = 1.647; ergo alpha is not statistically significant


# In[20]:


#Ask the model to project forwards
#Create a test dataframe X_test with years 2020-2050
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test



# In[21]:


X_test = test[['year','age','gender_female','gender_male','denom','intercept']]

X_test['age_sq'] = X_test['age']**2

X_test



# In[22]:


pois_predictions = result_year_age_gender.get_prediction(X_test[["intercept","year","age","age_sq","gender_male"]], 
                                                        )

pois_predictions_summary = pois_predictions.summary_frame()
print(pois_predictions_summary)


# In[23]:


#Add the predictions back into the test dataframe 

X_test['UKSP_rate_pred'] = pois_predictions_summary['mean']
X_test['UKSP_rate_lower'] = pois_predictions_summary['mean_ci_lower']
X_test['UKSP_rate_upper'] = pois_predictions_summary['mean_ci_upper']

X_test


# In[24]:


#Calculate absolute no.s of PD cases in the UK

X_test['Abs_case_midpoint'] = X_test['denom'] * X_test['UKSP_rate_pred'] / 100000
X_test['Abs_case_lower'] = X_test['denom'] * X_test['UKSP_rate_lower'] / 100000
X_test['Abs_case_upper'] = X_test['denom'] * X_test['UKSP_rate_upper'] / 100000

X_test


# In[25]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250328_PD_Inc_17yrs.csv')


# # Spline Poisson model

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')

df['uksp_rate'] = (df['pd'] / df['denom']) * 100000

df


# In[20]:


pdf=df[['year','age','gender','denom','pd']]

#pdf['year_sq'] = pdf['year']**2

pdf=pdf[pdf['year'] < 2020]

pdf


# In[21]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender','denom','pd'])
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age','gender'], drop_first=True)
#    .assign(intercept = 1)
    .sort_values(['year','gender_male',"age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5"])
    .reset_index(drop=True)
    )

y = X["pd"]

X


# In[22]:


#Create age band label
age_band_map = {
    "age_22.5": "20-24", "age_27.5": "25-29", "age_32.5": "30-34", "age_37.5": "35-39", "age_42.5": "40-44", "age_47.5": "45-49", "age_52.5": "50-54", "age_57.5": "55-59","age_62.5":"60-64","age_67.5":"65-69","age_72.5":"70-74","age_77.5":"75-79","age_82.5":"80-84","age_87.5":"85-89","age_92.5":"90-94","age_97.5":"95+"
}
age_band_cols = [col for col in X.columns if col.startswith("age_")]
X["age_band_label"] = X[age_band_cols].idxmax(axis=1).map(age_band_map)
X


# In[23]:


#Add in interaction terms
for col in age_band_cols:
    X[f"{col}_x_gender"] = X[col] * X["gender_male"]
    
X.info()


# In[24]:


#Spline basis functions
spline_manual = dmatrix("bs(year, knots=[2010.5], degree=3, include_intercept=False)", data=X, return_type='dataframe')
spline_auto = dmatrix("bs(year, df=6, degree=3, include_intercept=False)", data=X, return_type = 'dataframe')
piecewise = dmatrix("0 + cr(year, knots=[2010.5])", data=X, return_type = 'dataframe')


# In[25]:


#Build predictor sets
X_linear = X[['year','gender_male'] + age_band_cols + [f"{col}_x_gender" for col in age_band_cols]]
X_spline_manual = pd.concat([spline_manual, X[["gender_male"] + age_band_cols + [f"{col}_x_gender" for col in age_band_cols]]], axis=1)
X_spline_auto = pd.concat([spline_auto, X[["gender_male"] + age_band_cols + [f"{col}_x_gender" for col in age_band_cols]]], axis=1)
X_piecewise = pd.concat([piecewise, X[["gender_male"] + age_band_cols + [f"{col}_x_gender" for col in age_band_cols]]], axis=1)

#and offset term
offset = np.log(X["denom"] / 100000)



# In[9]:


#(Optional) Data prep checks

#Full rank check
np.linalg.matrix_rank(X_piecewise.values), X_piecewise.shape[1]
#print(X_piecewise.columns.tolist())

#NaNs and infs
#print(X_piecewise.isnull().sum())
#print(np.isinf(X_piecewise).sum())
#print(np.isnan(offset).sum(), np.isinf(offset).sum())


# In[26]:


#Fit models

model_linear = sm.GLM(y, X_linear, family=sm.families.Poisson(), offset=offset).fit()
model_spline_manual = sm.GLM(y, X_spline_manual, family=sm.families.Poisson(), offset=offset).fit()
model_spline_auto = sm.GLM(y, X_spline_auto, family=sm.families.Poisson(), offset=offset).fit()
model_piecewise = sm.GLM(y, X_piecewise, family=sm.families.Poisson(), offset=offset).fit()


# In[27]:


print(model_spline_manual.summary())


# In[28]:


#Compare AICs
print("AICs:")
print("Linear:", model_linear.aic)
print("Spline (manual):", model_spline_manual.aic)
print("Spline (auto):", model_spline_auto.aic)
print("Piecewise:", model_piecewise.aic)


# In[29]:


#Extract knots from spline_auto
print("Knots (auto):", spline_auto.design_info.describe())


# In[30]:


fitted_values = model_spline_manual.fittedvalues
results_table = pd.DataFrame({'y':y, 'Fitted Values': fitted_values})
sum_row = pd.DataFrame(results_table.sum(numeric_only=True)).T
sum_row.index = ['Total']
results_table = pd.concat([results_table, sum_row])
print(results_table)


# In[31]:


#Plot fitted values against observed data
plt.plot(y, model_spline_manual.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[32]:


#Plot fitted values against residuals

residuals = y - model_spline_manual.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(model_spline_manual.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[33]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[34]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = model_spline_manual.fittedvalues
X.to_csv('G:\My Drive\\17y_PD_Inc_fittedvalues_FINAL.csv')


# In[51]:


#Check overdispersion

def check_overdispersion(model):
    resid_dev = model.deviance
    df_resid = model.df_resid
    dispersion = resid_dev / df_resid
    print("Dispersion:", dispersion)
    return dispersion

check_overdispersion(model_spline_manual)


# In[52]:


# Negative binomial model

model_spline_manual_robust = sm.GLM(y, X_spline_manual, family=sm.families.NegativeBinomial(), offset=offset).fit()

print(model_spline_manual_robust.summary())


# # Use the nb model to generate forecast

# In[53]:


#Use the model created to make predictions

future_df=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')

future_df = future_df[['year','age','gender_male','denom']]

future_df


# In[54]:


#Prepare data for modelling

future_X = (future_df
    .groupby(['year','age','gender_male','denom'])
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age'], drop_first=True)
#    .assign(intercept = 1)
    .sort_values(['year','gender_male',"age_22.5","age_27.5","age_32.5","age_37.5","age_42.5","age_47.5","age_52.5","age_57.5","age_62.5","age_67.5","age_72.5","age_77.5","age_82.5","age_87.5","age_92.5","age_97.5"])
    .reset_index(drop=True)
    )

future_X


# In[55]:


#Create age band label
age_band_map = {
    "age_22.5": "20-24", "age_27.5": "25-29", "age_32.5": "30-34", "age_37.5": "35-39", "age_42.5": "40-44", "age_47.5": "45-49", "age_52.5": "50-54", "age_57.5": "55-59","age_62.5":"60-64","age_67.5":"65-69","age_72.5":"70-74","age_77.5":"75-79","age_82.5":"80-84","age_87.5":"85-89","age_92.5":"90-94","age_97.5":"95+"
}
age_band_cols = [col for col in future_X.columns if col.startswith("age_")]
future_X["age_band_label"] = future_X[age_band_cols].idxmax(axis=1).map(age_band_map)
future_X


# In[56]:


#Add in interaction terms
for col in age_band_cols:
    future_X[f"{col}_x_gender"] = future_X[col] * future_X["gender_male"]
    
future_X.info()


# In[57]:


X.info()


# In[58]:


#Prepare prediction set
combined = pd.concat([X, future_X], ignore_index=True)
combined_spline = dmatrix("bs(year, knots=[2010.5], degree=3, include_intercept=False)", data=combined, return_type='dataframe')
#combined_spline = dmatrix("bs(year, df=6, degree=3, include_intercept=False)", data=combined, return_type='dataframe')
X_pred = pd.concat([combined_spline, combined[["gender_male"] + age_band_cols + [f"{col}_x_gender" for col in age_band_cols]]], axis=1)
offset_pred = np.log(combined["denom"] / 100000)


# In[59]:


#Clip linear predictor values to prevent underflow of lower c.i.s
#linpred = model_spline_manual_robust.predict(X_pred, offset=offset_pred, linear=True)
linpred = model_spline_manual.predict(X_pred, offset = offset_pred, linear=True)
linpred = np.clip(linpred, a_min = -20, a_max = 20)
pred_mean = np.exp(linpred)
z = 1.96
std_err = model_spline_auto.bse.mean()
lower = np.exp(linpred - z * std_err)
upper = np.exp(linpred + z * std_err)
std_err


# In[60]:


#Add predictions
combined['pred_count'] = pred_mean
combined['pred_low'] = lower
combined['pred_high'] = upper
combined['pred_rate'] = (combined['pred_count'] / combined['denom']) * 100000
combined['rate_low'] = (combined['pred_low'] / combined['denom']) * 100000
combined['rate_high'] = (combined['pred_high'] / combined['denom']) * 100000
combined


# In[63]:


#Export to csv
combined.to_csv('G:\My Drive\\20250727_PD_Inc_2003-19_auto.csv')


# In[61]:


#Aggregate to annual totals
yearly_summary = (combined
    .groupby('year')
    .agg(
    total_count = ('pd', 'sum'),
    total_pred = ('pred_count','sum'),
    total_denom = ('denom', 'sum'))
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
yearly_summary


# In[62]:


#Plot total rate
plt.figure(figsize=(12,6))
plt.scatter(yearly_summary['year'], yearly_summary['total_count'], color="gray", alpha = 0.5, label = "Observed")
plt.plot(yearly_summary['year'], yearly_summary['total_pred'], label = 'Predicted', color='green')
#plt.fill_between(yearly_summary['year'], yearly_summary['pred_low'], yearly_summary['pred_high'], color='green', alpha=0.2)
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Title")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

