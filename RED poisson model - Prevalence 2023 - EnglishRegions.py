#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/llee/Downloads/PDprev_2023_aggregated_v1.txt', sep="\t")
df


# In[2]:


df = df.rename(columns={'PDpatients': 'Cases', 'ptcount': 'Denom', 'e2019_imd_5': 'imd'})

df['Cases'] = df['Cases'].fillna(0)
df['Denom'] = df['Denom'].fillna(0)

df


# In[3]:


df['Cases'] = df['Cases'].astype('float64')
df['Denom'] = df['Denom'].astype('float64')

df = df[df.Denom > 0]



# In[4]:


crosstab_counts = pd.crosstab(index=df['imd'], columns=df['region'], values=df['Cases'], aggfunc='sum')

print(crosstab_counts)


# In[5]:


#Age encoding
age_band_mapping = {
    '0-19': 1, '20-24': 2, '25-29': 3, '30-34': 4, '35-39': 5, '40-44': 6, 
    '45-49': 7, '50-54': 8, '55-59': 9, '60-64': 10, '65-69': 11, 
    '70-74': 12, '75-79': 13, '80-84': 14, '85-89': 15, '90-94': 16, '95+': 17
}


df['age_band'] = df['age_band'].map(age_band_mapping)


# In[6]:


#Gender encoding

df = df[df.gender != 'Other']

df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

df.info()


# In[7]:


print(df['ethnicity'].unique())


# In[8]:


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


# In[9]:


#IMD: Create two versions, ordinal and nominal 

imd_mapping = {
    '1.0': '1',
    '2.0': '2',
    '3.0': '3',
    '4.0': '4',
    '5.0': '5',
    'Unknown': 'Unknown'
}


df['imd'] = df['imd'].map(imd_mapping)

imd_dummies = pd.get_dummies(df['imd'], prefix = 'imd', drop_first = True)
df = pd.concat([df, imd_dummies], axis = 1)

df.info()


# In[10]:


urban_rural_mapping = {
    'Urban': 0,
    'Rural': 1,
}


df['urban_rural'] = df['urban_rural'].map(urban_rural_mapping)


# In[11]:


print(df['region'].unique())


# In[12]:


#Region encoding

#Remove Wales, Scotland and NI
df = df[df.region != 'Wales']
df = df[df.region != 'Scotland']
df = df[df.region != 'Northern Ireland']


#Sort so that North West is the reference
region_mapping = {
    'North East': 'b_North_East',
    'North West': 'a_North_West',
    'Yorkshire and The Humber': 'c_Yorkshire_and_the_Humber',
    'East Midlands': 'd_East_Midlands',
    'West Midlands': 'e_West_Midlands',
    'East of England': 'f_East_of_England',
    'London': 'g_London',
    'South East': 'h_South_East',
    'South West': 'i_South_West'
}

df['region'] = df['region'].map(region_mapping)

#Then create dummies
region_dummies = pd.get_dummies(df['region'], prefix = 'region', drop_first = True)
df = pd.concat([df, region_dummies], axis = 1)

df.info()


# In[13]:


df['Prevalence_Rate'] = (df['Cases'] / df['Denom'])
df.info()


# In[14]:


df.isnull().any().any() 


# In[15]:


df['Denom'].min()


# In[16]:


crosstab_counts = pd.crosstab(index=df['imd'], columns=df['region'], values=df['Cases'], aggfunc='sum')

print(crosstab_counts)


# ***FINAL MODEL STARTS HERE***

# In[17]:


#Build a Poisson model to predict the # cases instead using the log of the denominator as an offset
df['ln_denom'] = np.log(df['Denom'])

X = df[['age_band', 'gender', 'urban_rural', 'ethnicity_b_Asian', 'ethnicity_c_African_or_Caribbean','ethnicity_d_Mixed_or_Other','ethnicity_e_Unknown','imd_2','imd_3','imd_4','imd_5','imd_Unknown','region_b_North_East', 'region_c_Yorkshire_and_the_Humber', 'region_d_East_Midlands', 'region_e_West_Midlands', 'region_f_East_of_England', 'region_g_London', 'region_h_South_East', 'region_i_South_West']]
y = df['Cases']

X = sm.add_constant(X)

poisson_result = smf.glm("Cases ~ age_band + gender + urban_rural + ethnicity_b_Asian + ethnicity_c_African_or_Caribbean + ethnicity_d_Mixed_or_Other + ethnicity_e_Unknown + imd_2 + imd_3 + imd_4 + imd_5 + imd_Unknown + region_b_North_East + region_c_Yorkshire_and_the_Humber + region_d_East_Midlands + region_e_West_Midlands + region_f_East_of_England + region_g_London + region_h_South_East + region_i_South_West",offset=df['ln_denom'],data=df,family=sm.families.Poisson()).fit()

print(poisson_result.summary())


# In[18]:


#Residual diagnostics

plt.scatter(poisson_result.fittedvalues, poisson_result.resid_pearson)
plt.axhline(0, color='red', linestyle = '--')
plt.xlabel('Fitted values')
plt.ylabel('Pearson residuals')
plt.title('Residuals vs Fitted Values (Poisson)')
plt.show()


# In[19]:


#Check if residuals are symmetrical

plt.hist(poisson_result.resid_pearson, bins= 30, edgecolor = 'black')
plt.title('Histogram of Pearson residuals (Poisson)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[20]:


fig = sm.qqplot(poisson_result.resid_deviance, line='45', fit=True)
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()


# In[21]:


plt.scatter(poisson_result.fittedvalues, df['Cases'])
plt.plot(y, y, '--', label='y = x')
plt.xlabel('Fitted values')
plt.ylabel('Actual cases')
plt.title('Fitted Values vs Actuals (Poisson)')
plt.show()


# In[22]:


#Check for overdispersion
residual_deviance = poisson_result.deviance
residual_df = poisson_result.df_resid
dispersion_param = residual_deviance / residual_df

print("Deviance-based dispersion parameter:", dispersion_param)


# In[23]:


rate_ratios = np.exp(poisson_result.params)
print(rate_ratios)


# In[24]:


#Obtain more decimal points on p-values
p_values_extended = poisson_result.pvalues.round(5)
print(p_values_extended)


# In[25]:


coefficients = poisson_result.params
conf_int = poisson_result.conf_int()
conf_int.columns = ['2.5%', '97.5%']  # Renaming columns for clarity
rate_ratios = np.exp(coefficients)
conf_int_rate_ratios = np.exp(conf_int)

# Displaying coefficients and their 95% CI
print("\nCoefficients and 95% CI:")
coefficients_df = pd.DataFrame({
    "Coefficient": coefficients,
    "2.5% CI (Coeff)": conf_int["2.5%"],
    "97.5% CI (Coeff)": conf_int["97.5%"]
})
print(coefficients_df)

# Displaying Odds Ratios and their 95% CI
print("\nRate Ratios and 95% CI:")
rate_ratios_df = pd.DataFrame({
    "Rate Ratio": rate_ratios,
    "2.5% CI (OR)": conf_int_rate_ratios["2.5%"],
    "97.5% CI (OR)": conf_int_rate_ratios["97.5%"]
})
print(rate_ratios_df)


# In[26]:


print(poisson_result.fittedvalues)


# In[51]:


df['poisson_fitted'] = poisson_result.fittedvalues

df.to_csv('G:\My Drive\RED_region_pois.csv')


# In[27]:


import forestplot as fp

rate_ratios_df = rate_ratios_df.rename_axis("label").reset_index()

rate_ratios_df = rate_ratios_df[rate_ratios_df.label != 'Intercept']

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
    'age_band': 'Per 5 years',
    'region_b_North_East': 'North East',
    'region_c_Yorkshire_and_the_Humber': 'Yorkshire and the Humber',
    'region_d_East_Midlands': 'East Midlands',
    'region_e_West_Midlands': 'West Midlands',
    'region_f_East_of_England': 'East of England',
    'region_g_London': 'London',
    'region_h_South_East': 'South East',
    'region_i_South_West': 'South West'
}


rate_ratios_df['label'] = rate_ratios_df['label'].map(label_mapping)

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
    'Per 5 years': 'Age band',
    'North East': 'English Region (compared to North West)',
    'Yorkshire and the Humber': 'English Region (compared to North West)',
    'East Midlands': 'English Region (compared to North West)',
    'West Midlands': 'English Region (compared to North West)',
    'East of England': 'English Region (compared to North West)',
    'London': 'English Region (compared to North West)',
    'South East': 'English Region (compared to North West)',
    'South West': 'English Region (compared to North West)'
}

rate_ratios_df['group'] = rate_ratios_df['label'].map(group_mapping)

rate_ratios_df.info()

rate_ratios_df


# In[28]:


rate_ratios_df['Rate_summary'] = round(rate_ratios_df["Rate Ratio"],2)
rate_ratios_df['LL_summary'] = round(rate_ratios_df["2.5% CI (OR)"],2)
rate_ratios_df['UL_summary'] = round(rate_ratios_df["97.5% CI (OR)"],2)

rate_ratios_df['Rate_summary'] = rate_ratios_df['Rate_summary'].astype(str)
rate_ratios_df['LL_summary'] = rate_ratios_df['LL_summary'].astype(str)
rate_ratios_df['UL_summary'] = rate_ratios_df['UL_summary'].astype(str)

def pad_string_with_zeros(s):
    return s.ljust(4, '0') if len(s) < 4 else s

# Apply the function to each row in the column
rate_ratios_df['Rate_summary'] = rate_ratios_df['Rate_summary'].apply(pad_string_with_zeros)
rate_ratios_df['LL_summary'] = rate_ratios_df['LL_summary'].apply(pad_string_with_zeros)
rate_ratios_df['UL_summary'] = rate_ratios_df['UL_summary'].apply(pad_string_with_zeros)

rate_ratios_df['Summarised'] = rate_ratios_df['Rate_summary'] + " (" + rate_ratios_df["LL_summary"] + " to " + rate_ratios_df["UL_summary"] + ")"


rate_ratios_df                                                                                    


# In[30]:


filtered_df = rate_ratios_df[rate_ratios_df['group'] == 'English Region (compared to North Wesst)']
columns_to_print = ['label', 'Summarised']
print(filtered_df[columns_to_print])


# In[31]:


forest = fp.forestplot(rate_ratios_df, 
              estimate = "Rate Ratio", 
              ll="2.5% CI (OR)", hl="97.5% CI (OR)", 
              varlabel = "label", 
#              ylabel = "Rate ratio (95% Confidence interval)", 
              xlabel = "Rate ratio",
              logscale = False,
              color_alt_rows=True,
              rightannote = ["Summarised"],
              right_annoteheaders = ["Rate ratio (95% Confidence interval)"],
              ci_report = False,
              flush = True,
              xticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
              axvline = (1),
              groupvar = "group",
              group_order=["Age band", "Sex (compared to Female)", "Ethnicity (compared to White)", "Dwelling (compared to Urban)", "Deprivation (compared to IMD 5 least deprived)", "English Region (compared to North West)"],
              sort = False,
              table = True,
              **{
                  "marker": "D",
                  "fontfamily": "Times New Roman",
#                  'fontfamily': 'serif'
              }
             )

plt.xlabel("Rate ratio", fontname="Times New Roman")
#plt.ylabel("Odds ratio (95% Confidence interval)", fontname="Times New Roman", labelpad = 20)
plt.xticks(fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
#plt.text(0.01, 0.99, 'Figure A', transform=plt.gca().transAxes, fontname="Times New Roman")

forest.axvline(1, linewidth = 1, color = '#808080')

#plt.subplot(2,1,2)


plt.savefig("C:/Users/llee/Downloads/RED rate ratios_prevalence23_EnglishRegions_20250509.png", dpi=600, bbox_inches="tight")


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

