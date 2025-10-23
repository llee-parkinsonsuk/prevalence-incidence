## !/usr/bin/env python
# coding: utf-8

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V1_20250313_Secular_trends_PREV_all_years.csv')
df


# In[83]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[84]:


print(df.groupby('gender')['other_7'].sum().reset_index(), " ")


# In[85]:


df.info()


#   # b. 17 years 2003-19, Other 7 Parkinsonisms

# In[156]:


#Aggregate to annual totals
df_agg = (df
    .groupby(['year'])[["other_7"]]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
df_agg


# In[181]:


#

pdf=df[['year','age_band','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2020]

pdf=pdf[pdf['gender'] == 'male'] #Build a model for each gender separately

pdf


# In[182]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age_band','ln_denom'])[["other_7"]]
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['age_band'])
    .assign(intercept = 1)
    .sort_values(['year',"age_band_0-19","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+"])
    .reset_index(drop=True)
    )

y = X.pop("other_7")


# In[183]:


#X['age_sq'] = X['age']**2
#X['age_cubed'] = X['age']**3
#X['age_ln'] = np.log(X['age'])
#X['age_sqrt'] = np.sqrt(X['age'])
#X['year_sq'] = X['year']**2
#X['year_cubed'] = X['year']**3
#X['year_ln'] = np.log(X['year'])
#X['year_sqrt'] = np.sqrt(X['year'])

X


# In[184]:


y


# In[153]:


#1st model: intercept only with no indicator variables

import statsmodels.api as sm

model_no_indicators_F = sm.GLM(y, X["intercept"],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_no_indicators_F = model_no_indicators_F.fit()
print(result_no_indicators_F.summary())


# In[154]:


#Plot fitted values against observed values 

import matplotlib.pyplot as plt

plt.plot(y, result_no_indicators_F.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[185]:


#2nd model: add in age bands and year

model_age_year_F = sm.GLM(y, X[["intercept","age_band_0-19","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+","year"]],
                            offset=X["ln_denom"],
                            family=sm.families.Poisson(),
                            )
result_age_year_F = model_age_year_F.fit()
print(result_age_year_F.summary())


# In[94]:


plt.plot(y, result_age_year_F.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[95]:


#Plot fitted values against residuals

residuals = y - result_age_year_F.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result_age_year_F.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[96]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[186]:


#Converting model parameters to odds ratios

coefficients = result_age_year_F.params
conf_int = result_age_year_F.conf_int()
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


# In[98]:


#TEST OF EQUIDISPERSION

# Calculate residual deviance
residual_deviance = result_age_year_F.deviance

# Calculate degrees of freedom
df_res = result_age_year_F.df_resid

residual_deviance

# Calculate the ratio
ratio = residual_deviance / df_res

# Display the ratio
print("Residual Deviance:", residual_deviance)
print("Degrees of Freedom:", df_res)
print("Residual Deviance to Degrees of Freedom Ratio:", ratio)


# In[147]:


#Use an OLS regression model to predict future values of person-years-at-risk (denominator)

rdf = df[df['gender'] == 'female'] #change per each model iteration

rdf = rdf[rdf['age_band'] == '95+'] #change per each model iteration

rdf = rdf[['year','denom']]

rdf = rdf[rdf['year'] < 2020]

rdf['intercept'] = 1

rdf


# In[148]:


PYAR_F_0_19 = sm.OLS(rdf[['denom']], rdf[["intercept","year"]])

result_PYAR_F_0_19 = PYAR_F_0_19.fit()
print(result_PYAR_F_0_19.summary())


# In[149]:


coefficients = result_PYAR_F_0_19.params
conf_int = result_PYAR_F_0_19.conf_int()
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


# In[196]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result_year_age_gender_plus.fittedvalues
X['deviance_resids'] = result_year_age_gender_plus.resid_deviance
X['other_7'] = y
X.to_csv('G:\My Drive\poisson_data11.csv')


# In[180]:


#Obtain the value of alpha for the negative binomial model

import statsmodels.formula.api as smf


pdf['LAMBDA'] = result_year_age_gender_plus.mu

'''
Then we add a new column to our dataframe, 
which is derived from the λ vector.
It will serve as the dependent variable for our auxiliary OLS regression
'''

pdf['AUX_OLS'] = pdf.apply(lambda x: ((x['other_7'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)

# Specify the aux. OLS model 
ols_expr = """AUX_OLS ~ LAMBDA - 1"""

# Fit the aux. OLS model
aux_olsr_results = smf.ols(ols_expr, pdf).fit()

# Print regression parameters
print(aux_olsr_results.params)


# In[181]:


#Is alpha statistically significant?
aux_olsr_results.tvalues


# In[189]:


#Build the negative binomial model using the value of alpha we have obtained

nb_model_year_age_gender = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                                  offset=X["ln_denom"],
                                  family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0]))
result_nb_year_age_gender = nb_model_year_age_gender.fit()

print(result_nb_year_age_gender.summary())


# In[190]:


#Comparison of NB model vs Poisson

#Log-Likelihood = -3907.7 vs -61017

#Likelihood ratio test
-2* (result_year_age_gender.llf - result_nb_year_age_gender.llf)


# In[191]:


#Deviance
print ("Deviance:", result_nb_year_age_gender.deviance)
print ("Pearson's chi-sq:", result_nb_year_age_gender.pearson_chi2)

#critical value of chisq at 1% significance with 578d.f. ~= approx. 656
#Deviance and Pearson's chi-sq both < 656 ~ NB model provides good fit overall


# In[192]:


plt.plot(y, result_nb_year_age_gender.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[193]:


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


# In[194]:


#Odds ratio for year is 1.007227; annual INcrease therefore
annual_chg = np.exp(coefficients['year']) - 1
print("Annual change in prevalence = ", annual_chg*100, "%")

period_chg = annual_chg * 17
print("Total change in prevalence over period = ", period_chg*100, "%")


# In[195]:


#Export actuals and model values to CSV for visualisation

X['nb_fittedvalues'] = result_nb_year_age_gender.fittedvalues
X['nb_deviance_resids'] = result_nb_year_age_gender.resid_deviance
X.to_csv('G:\My Drive\poisson_data10.csv')


# In[121]:


#Ask the two model to project forwards
#Create a test dataframe X_test with years 2019-2023
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test



# In[122]:


X_test = test[['year','age','gender_female','gender_male','denom','log_denom','intercept']]

X_test['age_sq'] = X_test['age']**2
X_test['age_cubed'] = X_test['age']**3
X_test['age_ln'] = np.log(X_test['age'])
X_test['age_sqrt'] = np.sqrt(X_test['age'])
X_test['year_sq'] = X_test['year']**2
X_test['year_cubed'] = X_test['year']**3
X_test['year_ln'] = np.log(X_test['year'])
X_test['year_sqrt'] = np.sqrt(X_test['year'])

X_test['log_denom'] = np.log(X_test['denom'])

X_test



# In[123]:


pois_predictions = result_year_age_gender_plus.get_prediction(X_test[["intercept","year","age","age_sq","age_cubed","gender_male"]], 
                                  #offset=X_test["log_denom"])
#nb2_predictions = result_nb_year_age_gender.get_prediction(X_test[["intercept","year","age","age_sq","gender_male"]], 
#                                  offset=X_test["log_denom"])
                                                        )

pois_predictions_summary = pois_predictions.summary_frame()
print(pois_predictions_summary)

#nb2_predictions_summary = nb2_predictions.summary_frame()
#print(nb2_predictions_summary)


# In[124]:


#Add the predictions back into the test dataframe 

X_test['UKSP_rate_pred'] = pois_predictions_summary['mean']
X_test['UKSP_rate_lower'] = pois_predictions_summary['mean_ci_lower']
X_test['UKSP_rate_upper'] = pois_predictions_summary['mean_ci_upper']
#X_test['NB_pred'] = nb2_predictions_summary['mean']
#X_test['NB_lower'] = nb2_predictions_summary['mean_ci_lower']
#X_test['NB_upper'] = nb2_predictions_summary['mean_ci_upper']

X_test


# In[72]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250321_Other7_Prev_2003-19_v5.csv')


# In[125]:


#Aggregate to annual totals
X_agg = (X_test
    .groupby(['year'])[["UKSP_rate_pred","UKSP_rate_lower",'UKSP_rate_upper']]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
X_agg


# In[71]:


#To create a comparison plot

predicted_poisson=X_agg['UKSP_rate_pred']
#predicted_nb = X_agg['NB_pred']
fig = plt.figure()
fig.suptitle('Predicted pwp counts')
pois_predicted, = plt.plot(X_agg.index, predicted_poisson, 'go-', label='Poisson predicted counts')
#nb2_predicted, = plt.plot(X_agg.index, predicted_nb, 'ro-', label='Negative binomial predicted counts')
plt.legend(handles=[pois_predicted])
plt.show()


# In[126]:


pip install pygam


# In[133]:


from pygam import LinearGAM, s, f
import numpy as np

# Assuming 'gender' is already encoded as 0s and 1s
X_gam = X[['year', 'age', 'gender_female']].values
y_gam = pdf['other7_rate'].values

# Specify the model
# s() is used for continuous variables to fit a spline
# f() is used for categorical variables
gam = LinearGAM(s(0) + s(1) + f(2)).fit(X_gam, y_gam)


print(gam.summary())


# In[134]:


from matplotlib import pyplot as plt

# Plot the model's partial dependence for each predictor
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, i], pdep)
    plt.plot(XX[:, i], confi, c='r', ls='--')
    plt.title(f'Partial Dependence for feature {i}')
    plt.show()


# In[141]:


new_X_gam = test[['year', 'age', 'gender_female']].values
predictions = gam.predict(new_X_gam)

test['gam_predictions'] = predictions

test


# In[142]:


#Aggregate to annual totals
test_agg = (test
    .groupby(['year'])[["gam_predictions"]]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
test_agg


# In[144]:


test_agg = (test
    .groupby(['age'])[["gam_predictions"]]
    .sum()
    .reset_index()
    .sort_values(['age'])
    .reset_index(drop=True)
    )
test_agg


# In[143]:


#Export to csv
test.to_csv('G:\My Drive\\20250321_Other7_Prev_2003-19_v6.csv')


# # Spline Regression

# In[197]:


pip install patsy


# In[238]:


X['age_centered'] = X['age'] - X['age'].mean()
X['age_ctr_sq'] = X['age_centered'] ** 2
X['age_ctr_cubed'] = X['age_centered'] ** 3

X


# In[239]:


#Spline regression
import numpy as np
import pandas as pd
from patsy import dmatrix
import statsmodels.api as sm

X_spl = X[['age_centered', 'age_ctr_sq', 'age_ctr_cubed', 'year', 'gender_male']]

# Create spline for 'year' with a knot at 2015.5
# Using dmatrix from patsy to create design matrices
transformed_year = dmatrix("bs(year, knots=(2015.5,), degree=3, include_intercept=False)",
                            {"year": X_spl.year}, return_type='dataframe')

# Add 'age' and 'gender_male' directly
X_transformed = pd.concat([X_spl[['age_centered', 'age_ctr_sq', 'age_ctr_cubed', 'gender_male']], transformed_year], axis=1)

# Add a constant to the model
X_transformed = sm.add_constant(X_transformed)

# Dependent variable
y = pdf['other_7']

# Fit the model
model = sm.OLS(y, X_transformed).fit()

# Summary
print(model.summary())


# In[240]:


#Check collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_coll = X_spl[['age_centered', 'age_ctr_sq', 'age_ctr_cubed', 'year', 'gender_male']]  
X_coll = add_constant(X_coll)  

# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_coll.values, i) for i in range(X_coll.shape[1])]
vif["features"] = X_coll.columns

print(vif)


# In[241]:


# Generate a range of values for 'year' for plotting
year_range = np.linspace(X_spl['year'].min(), X_spl['year'].max(), 100)

# Create a design matrix for the year_range
X_year_spline = dmatrix("bs(year, knots=(2015.5,), degree=3, include_intercept=False)",
                        {"year": year_range}, return_type='dataframe')

# Add constant and other predictors at their mean values (for simplicity)
X_pred = sm.add_constant(pd.concat([pd.DataFrame({'age_centered': np.mean(X_spl['age_centered']), 'age_ctr_sq': np.mean(X_spl['age_ctr_sq']), 'age_ctr_cubed': np.mean(X_spl['age_ctr_cubed']), 'gender_male': np.mean(X_spl['gender_male'])}, index=X_year_spline.index), X_year_spline], axis=1))

# Predict using the model
y_pred = model.predict(X_pred)

# Plot
plt.plot(year_range, y_pred, label='Spline Effect of Year')
plt.xlabel('Year')
plt.ylabel('Predicted other_7')
plt.title('Spline Regression: Effect of Year on other_7')
plt.legend()
plt.show()


# In[242]:


# Get the fitted values from the model
fitted_values = model.fittedvalues

# Add the fitted values to the original DataFrame
X_spl['fitted_values'] = fitted_values


# In[243]:


X_spl


# In[244]:


#Export fitted values to csv
X_spl.to_csv('G:\My Drive\poisson_data12.csv')


# In[232]:


X_test


# In[236]:


X_test['age_centered'] = X_test['age'] - X_test['age'].mean()
X_test['age_ctr_sq'] = X_test['age_centered'] ** 2

X_test_spl = X_test[['age_centered','age_ctr_sq','gender_male','year']]

transformed_year_test = dmatrix("bs(year, knots=(2015.5,), degree=3, include_intercept=False)", {"year": X_test_spl.year}, return_type='dataframe')

# Drop the original 'year' column from X_test to avoid duplication
X_test_spl = X_test_spl.drop(columns=['year'])

# Concatenate the transformed_year_test with the rest of X_test
# Make sure to align columns correctly as they were in the training dataset
X_test_transformed = pd.concat([X_test_spl[['age_centered','age_ctr_sq','gender_male']], transformed_year_test], axis=1)

# Now add the constant term to X_test_transformed
X_test_transformed = add_constant(X_test_transformed, has_constant='add')

X_test_transformed


# In[237]:


#Make predictions
predictions = model.predict(X_test_transformed)
X_test_transformed['spline_pred'] = predictions


# # FINAL MODEL: Use simple OLS regression from the 2013-19 data

# In[1]:


#FINAL MODEL: Use simple linear regression from the 2016-19 data

import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V1_20250313_Secular_trends_PREV_all_years.csv')
df


# In[2]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['age_sq'] = df['age']**2

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[3]:


#

pdf=df[['year','age_band','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2020]

pdf=pdf[pdf['year'] > 2012] 

pdf


# In[4]:


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


# In[6]:


#Build the OLS model
import statsmodels.api as sm

model = sm.OLS(y, X[["intercept","year","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+","gender_male"]])

result = model.fit()
print(result.summary())


# In[8]:


import matplotlib.pyplot as plt
plt.plot(y, result.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[9]:


#Plot fitted values against residuals

residuals = y - result.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[10]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[11]:


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


# In[12]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result.fittedvalues
X['pd'] = y
X.to_csv('G:\My Drive\Prev_Other_7_fitted.csv')


# In[165]:


#See what happens when we ask the model to project forwards
#Create a test dataframe X_test with years 2019-2023
test=pd.read_csv('C:/Users/llee/Downloads/secular_trends_test - to 2050.csv')
test


# In[173]:


X_test = test[['year','age','gender_female','gender_male','denom','log_denom','intercept']]

#X_test['age_sq'] = X_test['age']**2

X_test['log_denom'] = np.log(X_test['denom'])

X_test



# In[174]:


X_test = (X_test
    .pipe(pd.get_dummies, columns=['age'])
    )

X_test


# In[175]:


X_test.columns = ['year', "gender_female","gender_male","denom","log_denom","intercept","age_band_0-19","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+"]

X_test


# In[176]:


model_predictions = result.get_prediction(X_test[["intercept","year","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+","gender_male"]], 
                                                      )

model_predictions_summary = model_predictions.summary_frame()
print(model_predictions_summary)


# In[177]:


#Add the predictions back into the test dataframe 

X_test['case_cnt_pred'] = model_predictions_summary['mean']
X_test['case_cnt_lower'] = model_predictions_summary['mean_ci_lower']
X_test['case_cnt_upper'] = model_predictions_summary['mean_ci_upper']


# In[178]:


#Export to csv
X_test.to_csv('G:\My Drive\\20250402_Other7_Prev_Predictions.csv')


# In[179]:


#Aggregate to annual totals
X_agg = (X_test
    .groupby(['year'])[["case_cnt_pred","case_cnt_lower",'case_cnt_upper']]
    .sum()
    .reset_index()
    .sort_values(['year'])
    .reset_index(drop=True)
    )
X_agg


# # FINAL MODEL: Use simple OLS regression from the 2003-13 data

# In[13]:


import pandas as pd

df=pd.read_csv('C:/Users/llee/Downloads/V1_20250313_Secular_trends_PREV_all_years.csv')
df


# In[14]:


import numpy as np

df['ln_denom'] = np.log(df['denom'])

df['age_sq'] = df['age']**2

df['other_7'] = df['secondary']+df['msa']+df['vp']+df['psp']+df['cbs']+df['dlb']+df['dip']
df


# In[15]:


#

pdf=df[['year','age_band','gender','ln_denom','other_7']]

pdf=pdf[pdf['year'] < 2014]

pdf


# In[16]:


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


# In[17]:


#Build the OLS model
import statsmodels.api as sm

model = sm.OLS(y, X[["intercept","year","age_band_20-24","age_band_25-29","age_band_30-34","age_band_35-39","age_band_40-44","age_band_45-49","age_band_50-54","age_band_55-59","age_band_60-64","age_band_65-69","age_band_70-74","age_band_75-79","age_band_80-84","age_band_85-89","age_band_90-94","age_band_95+","gender_male"]])

result = model.fit()
print(result.summary())


# In[18]:


import matplotlib.pyplot as plt
plt.plot(y, result.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[19]:


#Plot fitted values against residuals

residuals = y - result.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[20]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[21]:


#Export actuals and model values to CSV for visualisation

X['fittedvalues'] = result.fittedvalues
X['other_7'] = y
X.to_csv('G:\My Drive\Prev_Other_7_fitted_Early.csv')


# In[ ]:




