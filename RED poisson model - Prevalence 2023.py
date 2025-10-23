#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/llee/Downloads/lrdata2152_LL_reduced_unknowns.csv')
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


#Any nulls remaining?
df.isnull().any().any() 


# ***FINAL MODEL STARTS HERE***

# In[11]:


#Build a Poisson model to predict the # cases using the log of the denominator as an offset
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','ethnicity_e_Unknown','imd_2','imd_3','imd_4','imd_5','imd_Unknown']]
y = df['Cases']

X = sm.add_constant(X)

#poisson_result = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=df['Denom']).fit()
poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + ethnicity_e_Unknown + imd_2 + imd_3 + imd_4 + imd_5 + imd_Unknown",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[12]:


#Residual diagnostics

plt.scatter(poisson_result.fittedvalues, poisson_result.resid_pearson)
plt.axhline(0, color='red', linestyle = '--')
plt.xlabel('Fitted values')
plt.ylabel('Pearson residuals')
plt.title('Residuals vs Fitted Values (Poisson)')
plt.show()


# In[13]:


#Check if residuals are symmetrical

plt.hist(poisson_result.resid_pearson, bins= 30, edgecolor = 'black')
plt.title('Histogram of Pearson residuals (Poisson)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[14]:


#Q-Q Plot of deviance residuals

fig = sm.qqplot(poisson_result.resid_deviance, line='45', fit=True)
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()


# In[15]:


#Plot fitted vs actuals

plt.scatter(poisson_result.fittedvalues, df['Cases'])
plt.plot(y, y, '--', label='y = x')
plt.xlabel('Fitted values')
plt.ylabel('Actual cases')
plt.title('Fitted Values vs Actuals (Poisson)')
plt.show()


# In[16]:


#Check for overdispersion
residual_deviance = poisson_result.deviance
residual_df = poisson_result.df_resid
dispersion_param = residual_deviance / residual_df

print("Deviance-based dispersion parameter (Ordinal model):", dispersion_param)


# In[17]:


odds_ratios = np.exp(poisson_result.params)
print(odds_ratios)


# In[18]:


#Obtain more decimal points on p-values
p_values_extended = poisson_result.pvalues.round(5)
print(p_values_extended)


# In[19]:


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


# In[20]:


print(poisson_result.fittedvalues)


# In[23]:


df['poisson_fitted'] = poisson_result.fittedvalues

df.to_csv('G:\My Drive\RED_pois.csv')


# In[21]:


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


# In[22]:


odds_ratios_df['Odds_summary'] = round(odds_ratios_df["Odds Ratio"],2)
odds_ratios_df['LL_summary'] = round(odds_ratios_df["2.5% CI (OR)"],2)
odds_ratios_df['UL_summary'] = round(odds_ratios_df["97.5% CI (OR)"],2)

odds_ratios_df['Odds_summary'] = odds_ratios_df['Odds_summary'].astype(str)
odds_ratios_df['LL_summary'] = odds_ratios_df['LL_summary'].astype(str)
odds_ratios_df['UL_summary'] = odds_ratios_df['UL_summary'].astype(str)

def pad_string_with_zeros(s):
    return s.ljust(4, '0') if len(s) < 4 else s

# Apply the function to each row in the column
odds_ratios_df['Odds_summary'] = odds_ratios_df['Odds_summary'].apply(pad_string_with_zeros)
odds_ratios_df['LL_summary'] = odds_ratios_df['LL_summary'].apply(pad_string_with_zeros)
odds_ratios_df['UL_summary'] = odds_ratios_df['UL_summary'].apply(pad_string_with_zeros)

odds_ratios_df['Summarised'] = odds_ratios_df['Odds_summary'] + " (" + odds_ratios_df["LL_summary"] + " to " + odds_ratios_df["UL_summary"] + ")"


odds_ratios_df                                                                                    


# In[34]:


forest = fp.forestplot(odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
#              ylabel = "Rate ratio (95% Confidence interval)", 
              xlabel = "Rate ratio",
              logscale = True,
              color_alt_rows=True,
              rightannote = ["Summarised"],
              right_annoteheaders = ["Rate ratio (95% Confidence interval)"],
              ci_report = False,
              flush = True,
              xticks=[0.5, 0.75, 1.0, 1.5, 2.0],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Sex (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)"],
              sort = False,
              table = True,
              **{
                  "marker": "D",
                  "fontfamily": "Times New Roman",
#                  'fontfamily': 'serif'
              }
             )

plt.xlabel("Odds ratio", fontname="Times New Roman")
#plt.ylabel("Odds ratio (95% Confidence interval)", fontname="Times New Roman", labelpad = 20)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
#plt.text(0.01, 0.99, 'Figure A', transform=plt.gca().transAxes, fontname="Times New Roman")

forest.axvline(1, linewidth = 1, color = '#808080')

#plt.subplot(2,1,2)


plt.savefig("C:/Users/llee/Downloads/RED odds ratios_prevalence23_20250428.svg", dpi=600, bbox_inches="tight")


# In[35]:


inc_odds_ratios_df=pd.read_csv('G:\My Drive\\20250404_RED_Incidence_chartdata.csv')


inc_odds_ratios_df['Odds_summary'] = round(inc_odds_ratios_df["Odds Ratio"],2)
inc_odds_ratios_df['LL_summary'] = round(inc_odds_ratios_df["2.5% CI (OR)"],2)
inc_odds_ratios_df['UL_summary'] = round(inc_odds_ratios_df["97.5% CI (OR)"],2)

inc_odds_ratios_df['Odds_summary'] = inc_odds_ratios_df['Odds_summary'].astype(str)
inc_odds_ratios_df['LL_summary'] = inc_odds_ratios_df['LL_summary'].astype(str)
inc_odds_ratios_df['UL_summary'] = inc_odds_ratios_df['UL_summary'].astype(str)

def pad_string_with_zeros(s):
    return s.ljust(4, '0') if len(s) < 4 else s

# Apply the function to each row in the column
inc_odds_ratios_df['Odds_summary'] = inc_odds_ratios_df['Odds_summary'].apply(pad_string_with_zeros)
inc_odds_ratios_df['LL_summary'] = inc_odds_ratios_df['LL_summary'].apply(pad_string_with_zeros)
inc_odds_ratios_df['UL_summary'] = inc_odds_ratios_df['UL_summary'].apply(pad_string_with_zeros)

inc_odds_ratios_df['Summarised'] = inc_odds_ratios_df['Odds_summary'] + " (" + inc_odds_ratios_df["LL_summary"] + " to " + inc_odds_ratios_df["UL_summary"] + ")"


inc_odds_ratios_df     


# In[36]:


forest = fp.forestplot(odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
              #ylabel = "Odds ratio (95% Confidence interval)", 
              xlabel = "Rate ratio",
              logscale = True,
              color_alt_rows=True,
              rightannote = ["Summarised"],
              right_annoteheaders = ["Rate ratio (95% Confidence interval)"],
              ci_report = False,
              flush = True,
              xticks=[0.5,0.75,1.0,1.5,2.0],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Sex (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)"],
              sort = False,
              table = True,
              **{
                  "marker": "D",
                  "fontfamily": "Times New Roman",
#                  'fontfamily': 'serif'
              }
             )

plt.xlabel("Odds ratio", fontname="Times New Roman")
#plt.ylabel("Odds ratio (95% Confidence interval)", fontname="Times New Roman", labelpad = 20)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
#plt.text(0.01, 0.99, 'Figure A', transform=plt.gca().transAxes, fontname="Times New Roman")

forest.axvline(1, linewidth = 1, color = '#808080')

#plt.savefig("C:/Users/llee/Downloads/RED odds ratios_prevalence23_20250404.svg", dpi=300, bbox_inches="tight")


# In[40]:


forest2 = fp.forestplot(inc_odds_ratios_df, 
              estimate = "Odds Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
              #ylabel = "Odds ratio (95% Confidence interval)", 
              xlabel = "Rate ratio",
              logscale = True,
              color_alt_rows=True,
              rightannote = ["Summarised"],
              right_annoteheaders = ["Rate ratio (95% Confidence interval)"],
              ci_report = False,
              flush = True,
              xticks=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Sex (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)"],
              sort = False,
              table = True,
              **{
                  "marker": "D",
                  "fontfamily": "Times New Roman",
#                  'fontfamily': 'serif'
              }
             )

plt.xlabel("Odds ratio", fontname="Times New Roman")
#plt.ylabel("Odds ratio (95% Confidence interval)", fontname="Times New Roman", labelpad = 20)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
#plt.text(0.01, 0.99, 'Figure A', transform=plt.gca().transAxes, fontname="Times New Roman")

forest2.axvline(1, linewidth = 1, color = '#808080')

plt.savefig("C:/Users/llee/Downloads/RED odds ratios_incidence23_20250428.svg", dpi=300, bbox_inches="tight")


# In[163]:


pip install svg_stack


# In[173]:


#Stitch the two forest plots together as an SVG formatted image file
import svg_stack as ss

doc = ss.Document()

layout1 = ss.VBoxLayout()
layout1.addSVG('C:/Users/llee/Downloads/RED odds ratios_incidence23_WITH_TITLE_20250404.svg',alignment=ss.AlignTop|ss.AlignHCenter)
layout1.addSVG('C:/Users/llee/Downloads/RED odds ratios_prevalence23_WITH_TITLE_20250404.svg',alignment=ss.AlignCenter)

doc.setLayout(layout1)

doc.save('C:/Users/llee/Downloads/qt_api_test3.svg')


# In[18]:


#Add interaction terms
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','ethnicity_e_Unknown','imd_2','imd_3','imd_4','imd_5','imd_Unknown']]
y = df['Cases']

X = sm.add_constant(X)

#poisson_result = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=df['Denom']).fit()
poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + ethnicity_e_Unknown + imd_2 + imd_3 + imd_4 + imd_5 + imd_Unknown + age_band*urban_rural + age_band*gender + gender*urban_rural + ethnicity_b_Asian*urban_rural + ethnicity_c_African_or_Caribbean*urban_rural + ethnicity_b_Asian*age_band + ethnicity_c_African_or_Caribbean*age_band",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[19]:


odds_ratios = np.exp(poisson_result.params)

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

