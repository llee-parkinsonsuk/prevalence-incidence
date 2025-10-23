#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/llee/Downloads/V2_20250727_Secular_trends_INC_all_years.csv')

df['ln_denom'] = np.log(df['denom'])

df


# In[116]:


#
pdf=df[['year','age','gender','ln_denom','denom','secondary']] #Update for each condition (pd, msa, psp etc.)

pdf


# In[117]:


#Prepare data for modelling

X = (pdf
    .groupby(['year','age','gender','ln_denom'])[["secondary"]] #Update for each condition (pd, msa, psp etc.)
    .sum()
    .reset_index()
    .pipe(pd.get_dummies, columns=['gender'])
    .assign(intercept = 1)
    .sort_values(['year','age','gender_female'])
    .reset_index(drop=True)
    )

y = X.pop("secondary") #Update for each condition (pd, msa, psp etc.)


# In[118]:


X['age_sq'] = X['age']**2
X


# In[119]:


y


# In[120]:


#Poisson model

import statsmodels.api as sm
model = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                                  offset=X["ln_denom"],
                                  family=sm.families.Poisson(),
                 )
result_model = model.fit()

print(result_model.summary())


# In[121]:


#Obtain more decimal points on p-values
p_values_extended = result_model.pvalues.round(10)
print(p_values_extended)


# In[122]:


#Plot fitted values against observed values 

import matplotlib.pyplot as plt

plt.plot(y, result_model.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[123]:


#Plot fitted values against residuals

residuals = y - result_model.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result_model.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[124]:


#Q-Q plot
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[125]:


#Converting model parameters to odds ratios

coefficients = result_model.params
conf_int = result_model.conf_int()
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


# In[126]:


#TEST OF EQUIDISPERSION

# Get model residual deviance
residual_deviance = result_model.deviance

# Calculate degrees of freedom
df_res = result_model.df_resid

# Calculate the ratio
ratio = residual_deviance / df_res

# Display the ratio
print("Residual Deviance:", residual_deviance)
print("Degrees of Freedom:", df_res)
print("Residual Deviance to Degrees of Freedom Ratio:", ratio) #Values <1: underdispersion; >1: overdispersion, try NegBinomial


# In[127]:


nb_model = sm.GLM(y, X[["intercept","year","age","age_sq","gender_male"]],
                                  offset=X["ln_denom"],
                                  family=sm.families.NegativeBinomial(),
                 )
result_nb_model = nb_model.fit()

print(result_nb_model.summary())


# In[128]:


#Obtain more decimal points on p-values
p_values_extended = result_nb_model.pvalues.round(10)
print(p_values_extended)


# In[129]:


#Plot fitted values against observed values 

plt.plot(y, result_nb_model.fittedvalues, 'o')
plt.plot(y, y, '--', label='y = x')
plt.ylabel("fitted value")
plt.xlabel("observed value")
plt.legend()
plt.show()


# In[130]:


#Plot fitted values against residuals

residuals = y - result_nb_model.fittedvalues

plt.figure(figsize=(8, 6)) 
plt.scatter(result_nb_model.fittedvalues, residuals, alpha=0.5)  
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()


# In[131]:


#Q-Q plot

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[132]:


#Converting model parameters to odds ratios

coefficients = result_nb_model.params
conf_int = result_nb_model.conf_int()
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


# In[81]:


#TEST OF EQUIDISPERSION

# Get model residual deviance
residual_deviance = result_nb_model.deviance

# Calculate degrees of freedom
df_res = result_nb_model.df_resid

# Calculate the ratio
ratio = residual_deviance / df_res

# Display the ratio
print("Residual Deviance:", residual_deviance)
print("Degrees of Freedom:", df_res)
print("Residual Deviance to Degrees of Freedom Ratio:", ratio) #Values <1: underdispersion; >1: overdispersion, try NegBinomial


# In[ ]:




