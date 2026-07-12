#!/usr/bin/env python
# coding: utf-8

# # Bayesian Logistic Regression Applied to Candidemia data

# The purpose of this notebook is to apply a Bayesian analysis to Candidemia dataset.

# In[4]:


#Import libraries 

import os
import pymc as pm
import arviz as az 
import bambi as bm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, 
confusion_matrix, accuracy_score, 
f1_score,precision_recall_curve)


# In[1]:


#Import functions to retrieve data from Google Drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

#Set authentification and dirve
gauth = GoogleAuth()
drive = GoogleDrive(gauth)

#Find the Dataset on GoogleDrive 
fileList = drive.ListFile({'q': "title = 'data_candidemia.csv' and trashed=false"}).GetList()
for file1 in fileList:
  print('title: %s, id: %s' % (file1['title'], file1['id']))


# In[8]:


#Get data from Google Drive

#Get the id from CSV file
file_id = fileList[0]['id']
file = drive.CreateFile({'id':file_id})
file.GetContentFile('data_candidemia.csv')

#Convert it to a DataFrame
df = pd.read_csv('data_candidemia.csv', engine='python')
#Print the first row
df.head(1)


# In[10]:


#Prepare the data

#Clone the data
data=df[:]

#Drop unnecessary columns
data = data.drop(['Tempo de Internação', 'Obito Precoce', 'Obito Tardio'], axis=1)

#Dummy coding for categorical features
for i in data.drop(['Idade', 'Tempo em UTI', 'TAT-Dias'], axis=1).columns:
    if len(set(df[i].values)) > 2:
        data = pd.get_dummies(data=data, columns=[i], dtype=int)

#New column's name
new_names = ['Age','Gender','Culture','Recent Abdominal Surgery (within 30 days)','Central Venous Device (CVC, PICC, Hickman, Broviac, Semi-Implantable)',
            'Time in ICU','Fasting Status','Parenteral Nutrition','Dialysis During Candidemia','Mechanical Ventilation','Antibiotic (Vancomycin)',
            'Broad-Spectrum Antibiotic','Prior Use of Azoles','Previous Bacteremia (30 days or at candidemia onset)','Prematurity','Malignancy',
            'Short Bowel','Benign Hematologic Diseases','Inborn Error of Immunity','Neurological Sequelae','Heart Disease', 'Others*',
            'Septic Shock - use of amines', 'Complete Antifungal Treatment in days','Duration of Candidemia <= 2 Days',
            'Duration of Candidemia 3-7 Days', 'Duration of Candidemia >8 Days', 'Duration of Candidemia - Unknown','No SEPSE qSOFA','SEPSE qSOFA',
            'SEPSE qSOFA Unknown','No Disseminated Candidiasis','Disseminated Candidiasis', 'Disseminated Candidiasis Unknown','No Antifungal Treatment',
            'Antifungal Treatment - Azoles','Antifungal Treatment - Amphotericin','Antifungal Treatment - Echinocandidae','Antifungal Treatment - Combination',
            'Catheter Not Removed','Catheter Removed','Catheter Removal Unknown','Catheter Not Removed in the First 3 Days of Candidemia',
            'Catheter Removed in the First 3 Days of Candidemia','Catheter Removed in the First 3 Days of Candidemia - Unknown',
            'Antifungal Response - FAILURE', 'Antifungal Response <72H', 'Antifungal Response 4-7 Days','Antifungal Response >7 Days', 'Antifungal Response - Unknown']

#Rename columns
for i,j in enumerate(data.drop(['Neutropenia'],axis=1).columns):
    data=data.rename({j:new_names[i]},axis=1)

#Create ICU Status colmun
icu=[]
for i in data['Time in ICU'].values:
    if i > 0:
        icu.append(1)
    else:
        icu.append(0)
#Insert it to the main dataframe
data['ICU']=icu

#Check that ICU was added
data.loc[0:3,['Gender', 'ICU']]


# # Extract Relevant Features

# In the last notebook we plot and check correlation among the features, on this notebook we are gonna jump straight to extract the relevant features.

# In[9]:


#Features that correlate with ICU status
data.drop(['Time in ICU', 'ICU'],axis=1).corrwith(data.ICU).sort_values(ascending=True).plot(kind= 'barh', figsize=(20,9))
plt.title('Correlation with ICU')
plt.show()

#Print the R2 coefficients
data.drop(['Time in ICU', 'ICU'],axis=1).corrwith(data.ICU).sort_values(ascending=True)


# As we can see on the outcomes, most of the features are weakly correlated with ICU status. To extract the relevant features we are gonna consider the ones with a positive correlation equal or higher than 0.1 and for negative correlation equal or smaller than -0.1.

# In[12]:


#Extract relevant features

#ICU correlation columnt
correlation_target = data.drop(['Time in ICU'],axis=1).corr().ICU

#Extract relevant features
relevant_feat= correlation_target[(correlation_target <= -0.1) | (correlation_target >= 0.1)]
relevant_feat.sort_values(ascending=True)


# # Variance Inflation Factor (VIF)

# In[12]:


#Let's create the function to apply VIF

#Import library
from sklearn.linear_model import LinearRegression

#Define the function
def calculate_vif(data, features):
    vif, tolerance= {},{}
    #Extract all the relevant features
    for feature in features:
        #Retrieve the relevant features
        X= [f for f in features if f != feature]

        #X contains all the predictors except one, y contains that predictor to be regressed against
        X,y= data[X], data[feature]

        #R2
        r2= LinearRegression().fit(X,y).score(X,y)

        #Tolerance 
        tolerance[feature] = 1 - r2

        #VIF
        vif[feature]= 1/1-tolerance[feature]
    #Return
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


# In[14]:


#Calculate the VIF
calculate_vif(data=data, features=relevant_feat.drop(['ICU'], axis=0).index)


# All features are perfectly independent, good sign to begging with the bayesian analysis.

# # Bayesian approach applied to relevant features

# Due to the large number of predictors relative to the limited number of subjects in the dataset, fitting a fully saturated multivariable logistic regression model is statistically inappropriate and may lead to unstable or non-identifiable parameter estimates. In this context, including all predictors with dummy variables simultaneously can inflate posterior uncertainty, reduce effective sample size, and generate divergences during sampling, indicating that the model is over-parameterized relative to the available information. For these reasons, it is necessary to limit the number of predictors and fit a more parsimonious model. 

# To avoid any issue while fitting the model, one dummy-coded variable it's gonna be chosen among the features with more than 2 categories. In this case, we are gonna keep:
# 
# - **Catheter Removed**
# - **Duration of Candidemia <=2 Days**
# - **Antifungal Response <72H**
# - **Antifungal Response - Unknown**
# - **Catheter Removed in the First 3 Days of Candidemia**

# ### Antifungal Response - FAILURE Case
# 
# Taking into account the last notebook where a classic Logistic Regression was applied, on that notebook we can notice that Antifungal Response - FAILURE has a problem with *quase-complete separation*, because of that this feature it's gonna be removed from all studies. Here is the crosstab that validates this information.

# In[149]:


#Crosstab Antifungal Response - FAILURE vs ICU status
pd.crosstab(data['Antifungal Response - FAILURE'],data.ICU)


# In[14]:


#Fit a Logistic Regression with a Bayesian approach
mu= 0
sigma= 1 #1 1.5

#Standarize Age
standarize_age= (data.Age - data.Age.mean())/ data.Age.std()
#Alternative:
#for i in data.Age:
#    standarize_age.append((i - np.mean(data.Age))/np.std(data.Age))

with pm.Model() as relevant_feat_model:

    #Priors with weakly informative normal distributions
    beta_0 = pm.Normal('beta_0', mu= mu, sigma= sigma)
    beta_Age = pm.Normal('beta_Age', mu= mu, sigma= sigma)
    beta_Culture = pm.Normal('beta_Culture', mu= mu, sigma= sigma)
    beta_Fasting_Status= pm.Normal('beta_Fasting_Status', mu= mu, sigma= sigma)
    beta_Parenteral_Nutrition = pm.Normal('beta_Parenteral_Nutrition', mu= mu, sigma= sigma)
    beta_Dialysis_During_Candidemia = pm.Normal('beta_Dialysis_During_Candidemia', mu= mu, sigma= sigma)
    beta_Mechanical_Ventilation = pm.Normal('beta_Mechanical_Ventilation', mu= mu, sigma= sigma)
    beta_Antibiotic = pm.Normal('beta_Antibiotic', mu= mu, sigma= sigma)
    beta_Broad_Spectrum_Antibiotic = pm.Normal('beta_Broad_Spectrum_Antibiotic', mu= mu, sigma= sigma)
    beta_Neutropenia = pm.Normal('beta_Neutropenia', mu= mu, sigma= sigma)
    beta_Prematurity = pm.Normal('beta_Prematurity', mu= mu, sigma= sigma)
    beta_Malignancy = pm.Normal('beta_Malignancy', mu= mu, sigma= sigma)
    beta_Benign_Hematologic_Diseases = pm.Normal('beta_Benign_Hematologic_Diseases', mu= mu, sigma= sigma)
    beta_Heart_Disease = pm.Normal('beta_Heart_Disease', mu= mu, sigma= sigma)
    beta_Septic_Shock = pm.Normal('beta_Septic_Shock', mu= mu, sigma= sigma)
    beta_Duration_Candidemia_less_2D = pm.Normal('beta_Duration_Candidemia_less_2D', mu= mu, sigma= sigma)
    beta_Duration_Candidemia_Unknown = pm.Normal('beta_Duration_Candidemia_Unknown', mu= mu, sigma= sigma)
    beta_No_SEPSE_qSOFA = pm.Normal('beta_No_SEPSE_qSOFA', mu= mu, sigma= sigma)
    beta_SEPSE_qSOFA = pm.Normal('beta_SEPSE_qSOFA', mu= mu, sigma= sigma)
    beta_Disseminated_Candidiasis = pm.Normal('beta_Disseminated_Candidiasis', mu= mu, sigma= sigma)
    beta_Azoles = pm.Normal('beta_Azoles', mu= mu, sigma= sigma)
    beta_Echinocandidae = pm.Normal('beta_Echinocandidae', mu= mu, sigma= sigma)
    beta_Catheter_Not_Removed = pm.Normal('beta_Catheter_Not_Removed', mu= mu, sigma= sigma)
    beta_Catheter_Removed = pm.Normal('beta_Catheter_Removed', mu= mu, sigma= sigma)
    beta_Catheter_Not_Removed_3Days = pm.Normal('beta_Catheter_Not_Removed_3Days', mu= mu, sigma= sigma)
    beta_Catheter_Removed_3Days = pm.Normal('beta_Catheter_Removed_3Days', mu= mu, sigma= sigma)
    beta_Antifungal_Response_FAILURE = pm.Normal('beta_Antifungal_Response_FAILURE', mu= mu, sigma= sigma)
    beta_Antifungal_Response_less72H = pm.Normal('beta_Antifungal_Response_less72H', mu= mu, sigma= sigma)
    beta_Antifungal_Response_Unknown = pm.Normal('beta_Antifungal_Response_Unknown', mu= mu, sigma= sigma)

    #Model fit
    p= pm.Deterministic('p', pm.math.sigmoid(beta_0 + beta_Age*standarize_age + beta_Culture*data.Culture + beta_Fasting_Status*data['Fasting Status']
                                                + beta_Parenteral_Nutrition*data['Parenteral Nutrition'] + beta_Dialysis_During_Candidemia*data['Dialysis During Candidemia']
                                                + beta_Mechanical_Ventilation*data['Mechanical Ventilation'] + beta_Antibiotic*data['Antibiotic (Vancomycin)']
                                                + beta_Broad_Spectrum_Antibiotic*data['Broad-Spectrum Antibiotic']
                                                + beta_Neutropenia*data['Neutropenia'] + beta_Prematurity*data.Prematurity + beta_Malignancy*data.Malignancy 
                                                + beta_Benign_Hematologic_Diseases*data['Benign Hematologic Diseases'] + beta_Heart_Disease*data['Heart Disease']
                                                + beta_Septic_Shock*data['Septic Shock - use of amines'] + beta_Duration_Candidemia_less_2D*data['Duration of Candidemia <= 2 Days']
                                                + beta_Duration_Candidemia_Unknown*data['Duration of Candidemia - Unknown'] #+ beta_No_SEPSE_qSOFA*data['No SEPSE qSOFA']
                                                + beta_SEPSE_qSOFA*data['SEPSE qSOFA'] + beta_Disseminated_Candidiasis*data['Disseminated Candidiasis'] + beta_Azoles*data['Antifungal Treatment - Azoles']
                                                + beta_Echinocandidae*data['Antifungal Treatment - Echinocandidae'] #+ beta_Catheter_Not_Removed*data['Catheter Not Removed']
                                                + beta_Catheter_Removed*data['Catheter Removed']#+ beta_Catheter_Not_Removed_3Days*data['Catheter Not Removed in the First 3 Days of Candidemia']
                                                + beta_Catheter_Removed_3Days*data['Catheter Removed in the First 3 Days of Candidemia'] #+ beta_Antifungal_Response_FAILURE*data['Antifungal Response - FAILURE']
                                                + beta_Antifungal_Response_less72H*data['Antifungal Response <72H'] + beta_Antifungal_Response_Unknown*data['Antifungal Response - Unknown']))


    #Likelihood
    pm.Bernoulli('ICU Admissions', p, observed=data.ICU)


# In[16]:


#Sampling
with relevant_feat_model:
    idata= pm.sample(draws= 2000, tune= 1000, target_accept=0.95, random_seed=42, idata_kwargs={'log_likelihood':True})
sum = az.summary(idata)


# In[18]:


#Print the summary
sum.iloc[0:29,:]


# In[38]:


#Relevant findings
relevant_ind=[]
for i in sum.index[0:29]:
    if not (sum.loc[i,'hdi_3%'] < 0) & (sum.loc[i,'hdi_97%'] > 0):
        relevant_ind.append(i)
sum.loc[relevant_ind,:]


# In[99]:


#Find the Posterior Odds Ratio

#Extract OR
or_samples, indexes=[],[]
for i in sum.iloc[0:29,:].index:
    if not (sum.loc[i,'hdi_3%'] < 0) & (sum.loc[i,'hdi_97%'] > 0):
        #Posterior samples of beta
        or_samples.append(np.exp(sum.loc[i,'mean']))
        #Indexes
        indexes.append(i)

#Find percentage
per=[]
for i in or_samples:
    per.append(round((i - 1)*100, 2))

#Copy dataframe
copy= sum.drop(['sd','mcse_mean','mcse_sd','ess_bulk','ess_tail'], axis=1).loc[indexes,:]

#Insert the new info to the dataframe
copy.insert(1, 'Posterior Odds Ratio', or_samples)
copy.insert(2, 'Percentage', per)

#Print relevant data
copy


# **Interpretation**
# 
# After analyzing the results, it is important to highlight that all the features/covariates converge well. The parameter *r_hat* validates this statement because all the features got a r_hat of 1.
# 
# - **Mechanical Ventilation**</br>
# *Posterior mean (log-OR) = 1.077*</br>
# *94 % HDI = 0.106 – 2.188*</br>
# *Posterior OR = exp(1.077) ≈ 2.94*</br>
# 
# After adjusting for all other covariates in the model, the odds of ICU admission among children with candidemia who required mechanical ventilation were an estimated 2.9 times the odds of ICU admission among those who did not require ventilation (94 % credible interval for the odds ratio: 1.11 – 8.91).
# Because the entire HDI is positive, we assign high posterior probability (> 97 %) to a beneficial (in this case, risk-increasing) association between mechanical ventilation and ICU admission.
# 
# - **Septic Shock – use of amines**</br>
# *Posterior mean (log-OR) = 1.482*</br>
# *94 % HDI = 0.256 – 2.738*</br>
# *Posterior OR = exp(1.482) ≈ 4.40*</br>
# 
# Children in septic shock treated with vaso-active amines exhibited an estimated 4.4-fold increase in the odds of ICU admission relative to children without this condition (94 % credible interval for the odds ratio: 1.29 – 15.46).
# The entire HDI lies above zero, indicating strong posterior evidence that septic shock with amine support is associated with a materially higher risk of ICU transfer in the paediatric candidemia cohort.
# 

# In[167]:


trace= az.plot_trace(idata,compact=False)


# According with the plot, the algorithm does converge.

# # Evaluation of the model

# ### Posterior Predictive Check (PPC)

# In[28]:


#Sample PPC
with relevant_feat_model:
    ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

#Plot the PPC
plt.style.use('arviz-colors')
pm.plot_ppc(idata)
#, num_pp_samples= 300, mean=False)


# The blue shaded region shows the kernel-density estimates of the ICU-admission rate across the posterior-replicated datasets; the solid black line marks the observed proportion in the real data. Because the black curve falls well inside the blue cloud, the model captures the empirical event rate and is considered well calibrated.

# ### Separation Plot

# In[72]:


#Graph the separation plot
az.plot_separation(idata=ppc, y="ICU Admissions",figsize=(12,1))


# **Interpretation of the Separation Plot**</br>
# 
# *Each bar* - one patient.</br>
# *Light blue bars* - predicted low probability of ICU admission.</br>
# *Dark blue bars* - actual ICU admissions (events), placed on the probability scale. </br>
# 
# - The first half of the plot is mostly light blue, with very few dark-blue event bars. This indicates that the model assigns low ICU risk to a large portion of the children, and—importantly—most of these patients indeed were not admitted to the ICU, leading to a good specificity.</br>
# 
# - Events represented by dark blue bars are concentrated toward the right side. This is exactly what good separation looks like: predicted high-risk patients include most of the real ICU admissions, meaning a good sensitivity in the high-risk region.</br>
# 
# - The far right of the plot is almost entirely dark blue indicating a subset of patients for whom the model assigns a very high predicted probability, and almost all of them did end up in the ICU. The model very confidently identifies the highest-risk group.</br>
# 
# - Some scattered dark-blue bars appear in the middle region. These are misclassified ICU admissions whose predicted probabilities were not extremely high. This is expected with small sample sizes and clinical heterogeneity.</br>
# 
# In summary, the model shows meaningful discriminative ability:</br>
# - ICU admissions tend to cluster where predicted probabilities are high.</br>
# - Misclassifications are present but not excessive.</br>
# - Given the small sample size and high dimensionality of the dataset, this level of separation is actually quite reasonable.</br>
# - The inclusion of only the predictors that showed meaningful correlation with ICU status seems to have improved separation by reducing model noise and instability.

# ### k̂ Parameter

# This section is dedicated to find influential observations that can alter the outcome of the model, a Leav One Out cross validation it's gonna be used to identify if there is any influential observatio through the arviz function *loo*.
# 
# According with the literature, an observation is considered influental when its K_hat parameter or pareto parameter is bigger than 0.7.

# In[16]:


# compute pointwise LOO
loo = az.loo(idata)

#Plot k values
az.plot_khat(loo.pareto_k)#, show_bins=True)


# As we can notice on the K_hat plot, all the observations are under 0.7, so we can claim there is no influential obersvations that could influence the model. There is only one point above 0.5, lets find that patient.

# In[18]:


#Find the subject with the highest influential observation

ax= az.plot_khat(loo.pareto_k.values.ravel()) #ravel puts all the pareto values in one single array
sorted_kappas = np.sort(loo.pareto_k.values.ravel())

#Find the observation where the kappa value exceeds the threshold
threshold= sorted_kappas[-1:]
ax.axhline(threshold, ls='--', color= 'gray')
influential_obs= data.reset_index()[loo.pareto_k.values >= threshold].index

for i in influential_obs:
    y= loo.pareto_k.values[i]
    ax.text(i, y + 0.007, str(i), ha= 'center', va='baseline')


# In[199]:


#Find the subject with the highest influential observation on the dataframe
data.iloc[influential_obs,:]


# In[66]:


#Print of the influential observation value
print('The highest kappa value is %s' % round(threshold[0],4))


# In[70]:


#Function to convert age from months to years and months
def age_converter(age):
    year = age // 12
    months = age % 12
    return year,months
year,month = age_converter(25)
print('The subject with the highest influential observations is %s years and %s month old.' % (year,month))


# The subject with the highest influential observation is a boy, he is 2 years old, and he was not admitted to the ICU. 
# 
# As we just mentioned before, for an influential observation to have a impact over the outcome of the model must have a kappa value greater than 0.7, in this case the influential value is around 0.5.

# In[60]:


#Find the subject on the Separation graph
import matplotlib.patheffects as pe

#Create the separation plot
ax= az.plot_separation(idata=ppc, y='ICU Admissions', figsize=(12,1))

#Set the y axis position
y= np.random.uniform(low= 0.1, high= 0.7, size= len(influential_obs))

#Marke the highest kappa observation
for x,y in zip(influential_obs, y):
    #Printing of the value
    text= str(x)
    x= x/len(data)
    #Create the mark
    ax.scatter(x=x, y=y, marker= '+', s=50, color='red', zorder=3)
    ax.text(x=x, y= y + 0.1, s=text, color= 'white', ha='center', va='bottom',
           path_effects= [pe.withStroke(linewidth=2, foreground= 'black')])


# One non-ICU patient appears in the extreme right tail of the separation plot. This indicates that, based on the predictors included in the model, this individual had a high posterior probability of ICU admission. However, the κ influence statistic for this observation was 0.51, substantially below the commonly used threshold of 0.7, indicating that the observation is not influential and does not distort the model.</br>
# The presence of high-probability non-events is expected in probabilistic models, particularly in heterogeneous clinical datasets where patients with severe markers may ultimately not require ICU admission. Therefore, this case reflects model uncertainty rather than a model misspecification or failure. This is a classical false positive, and some false positives are expected in any probabilistic model.

# ### Forest Plot

# In[45]:


#Graph the Forest plot
az.plot_forest(idata, combined=True, colors= 'slategray', var_names=sum.index[0:29], figsize=(6,9))
plt.axvline(x=0, c= 'red', linestyle='--',alpha = 0.4)
plt.savefig('FRB_relevant.jpg',bbox_inches='tight')


# The forest plot confirms the outcomes from the summary table, here we can see that the only features associated with ICU admissions are *Mechanical Ventilation* and *Septic Shock*, they both increase the odds of ICU admission.

# ### Confussion matrix

# In[89]:


#Import library
from sklearn.metrics import classification_report

#Posterior-mean probability for each patient
y_prob = idata.posterior['p'].mean(('chain','draw')).values

#Bayes optimal hard call
#Element-wise threshold at 0.5 → True/False.
y_pred= (y_prob > 0.5).astype(int)

#Confussion matrix
print('Confusion Matrix \n',confusion_matrix(data.ICU, y_pred))
print(classification_report(data.ICU, y_pred, digits=3))


# The model demonstrates strong discriminative performance for predicting ICU admission among pediatric patients with candidemia. It correctly identifies 88.7% of true ICU cases (high recall), while maintaining a precision of 84.6%, indicating few false alarms. Although performance for the non-ICU class is slightly lower (recall 72.2%), the model prioritizes correctly identifying high-risk patients, which is clinically desirable. Overall accuracy is 82.7%, and the F1-scores indicate balanced and robust classification for both classes.

# ### Area Under the Curve

# In[103]:


#Create a dictionary to get the parameters for AUC
pred_scores = dict(y_true=data.ICU, y_score=y_pred)
#Perform AUC
print(f'AUC = {roc_auc_score(**pred_scores):.4f}')


# The outcome indicates good discriminative ability. This means that, in approximately 80% of random patient pairs, the model assigns a higher predicted probability of ICU admission to the patient who was actually admitted. Given the clinical importance of detecting high-risk cases in pediatric candidemia, an AUC above 0.80 demonstrates that the model performs reliably and supports effective risk stratification. Combined with the high recall for ICU cases (0.887) and an overall accuracy of 82.7%, these results indicate that the model has strong predictive capacity with acceptable balance between sensitivity and specificity.

# # Bayesian approach applied to all features

# In this section, the same Bayesian analysis its gonna be applied to the same dataset but this time containing all the features. To avoid any problem while fitting the model a few dummy variables are gonna be taken as references, and as well as with the first model, the dummy variable *Antifungal Response - Failure* it's not gonnna be considered to avoid any separation problems.

# In[138]:


#Let's take a look on the features that weren't consider on the first model
print('Features not considered on the first model:')
print()
for i in data.columns:
    if i not in relevant_feat.index:
        print(i)


# The features taken as reference are:
# 
# - **Duration of Candidemia - Unknown**
# - **SEPSE qSOFA Unknown**
# - **Disseminated Candidiasis Unknown**
# - **No Antifungal Treatment**
# - **Catheter Removal Unknown**
# - **Catheter Removed in the First 3 Days of Candidemia - Unknown**
# - **Antifungal Response - FAILURE**

# In[101]:


#Fit a Logistic Regression with a Bayesian approach
mu= 0
sigma= 1 #1 1.5

#Standarize Age
standarize_age= (data.Age - data.Age.mean())/ data.Age.std()
#Alternative:
#for i in data.Age:
#    standarize_age.append((i - np.mean(data.Age))/np.std(data.Age))

#Standarize Complete antifungal treatment in days
catd = (data['Complete Antifungal Treatment in days'] - data['Complete Antifungal Treatment in days'].mean())/data['Complete Antifungal Treatment in days'].std()

with pm.Model() as full_model:

    #Priors with weakly informative normal distributions
    beta_0 = pm.Normal('beta_0', mu= mu, sigma= sigma)
    beta_Age = pm.Normal('beta_Age', mu= mu, sigma= sigma)
    beta_Culture = pm.Normal('beta_Culture', mu= mu, sigma= sigma)
    beta_Fasting_Status= pm.Normal('beta_Fasting_Status', mu= mu, sigma= sigma)
    beta_Parenteral_Nutrition = pm.Normal('beta_Parenteral_Nutrition', mu= mu, sigma= sigma)
    beta_Dialysis_During_Candidemia = pm.Normal('beta_Dialysis_During_Candidemia', mu= mu, sigma= sigma)
    beta_Mechanical_Ventilation = pm.Normal('beta_Mechanical_Ventilation', mu= mu, sigma= sigma)
    beta_Antibiotic = pm.Normal('beta_Antibiotic', mu= mu, sigma= sigma)
    beta_Broad_Spectrum_Antibiotic = pm.Normal('beta_Broad_Spectrum_Antibiotic', mu= mu, sigma= sigma)
    beta_Neutropenia = pm.Normal('beta_Neutropenia', mu= mu, sigma= sigma)
    beta_Prematurity = pm.Normal('beta_Prematurity', mu= mu, sigma= sigma)
    beta_Malignancy = pm.Normal('beta_Malignancy', mu= mu, sigma= sigma)
    beta_Benign_Hematologic_Diseases = pm.Normal('beta_Benign_Hematologic_Diseases', mu= mu, sigma= sigma)
    beta_Heart_Disease = pm.Normal('beta_Heart_Disease', mu= mu, sigma= sigma)
    beta_Septic_Shock = pm.Normal('beta_Septic_Shock', mu= mu, sigma= sigma)
    beta_Duration_Candidemia_less_2D = pm.Normal('beta_Duration_Candidemia_less_2D', mu= mu, sigma= sigma)
    beta_Duration_Candidemia_Unknown = pm.Normal('beta_Duration_Candidemia_Unknown', mu= mu, sigma= sigma)
    beta_No_SEPSE_qSOFA = pm.Normal('beta_No_SEPSE_qSOFA', mu= mu, sigma= sigma)
    beta_SEPSE_qSOFA = pm.Normal('beta_SEPSE_qSOFA', mu= mu, sigma= sigma)
    beta_Disseminated_Candidiasis = pm.Normal('beta_Disseminated_Candidiasis', mu= mu, sigma= sigma)
    beta_Azoles = pm.Normal('beta_Azoles', mu= mu, sigma= sigma)
    beta_Echinocandidae = pm.Normal('beta_Echinocandidae', mu= mu, sigma= sigma)
    beta_Catheter_Not_Removed = pm.Normal('beta_Catheter_Not_Removed', mu= mu, sigma= sigma)
    beta_Catheter_Removed = pm.Normal('beta_Catheter_Removed', mu= mu, sigma= sigma)
    beta_Catheter_Not_Removed_3Days = pm.Normal('beta_Catheter_Not_Removed_3Days', mu= mu, sigma= sigma)
    beta_Catheter_Removed_3Days = pm.Normal('beta_Catheter_Removed_3Days', mu= mu, sigma= sigma)
    beta_Antifungal_Response_FAILURE = pm.Normal('beta_Antifungal_Response_FAILURE', mu= mu, sigma= sigma)
    beta_Antifungal_Response_less72H = pm.Normal('beta_Antifungal_Response_less72H', mu= mu, sigma= sigma)
    beta_Antifungal_Response_Unknown = pm.Normal('beta_Antifungal_Response_Unknown', mu= mu, sigma= sigma)
    #New features:
    beta_Gender = pm.Normal('beta_Gender', mu= mu, sigma= sigma) 
    beta_Recent_Abdominal_Surgery = pm.Normal('beta_Recent_Abdominal_Surgery', mu= mu, sigma= sigma)
    beta_Central_Venous_Device = pm.Normal('beta_Central_Venous_Device', mu= mu, sigma= sigma)
    beta_Prior_Use_of_Azoles = pm.Normal('beta_Prior_Use_of_Azoles', mu= mu, sigma= sigma)
    beta_Previous_Bacteremia = pm.Normal('beta_Previous_Bacteremia', mu= mu, sigma= sigma)
    beta_Short_Bowel = pm.Normal('beta_Short_Bowel', mu= mu, sigma= sigma)
    beta_Inborn_Error_of_Immunity = pm.Normal('beta_Inborn_Error_of_Immunity', mu= mu, sigma= sigma)
    beta_Neurological_Sequelae = pm.Normal('beta_Neurological_Sequelae', mu= mu, sigma= sigma)
    beta_Complete_Antifungal_Treatment_in_days = pm.Normal('beta_Complete_Antifungal_Treatment_in_days', mu= mu, sigma= sigma)
    beta_Duration_of_Candidemia_3to7Days = pm.Normal('beta_Duration_of_Candidemia_3to7Days', mu= mu, sigma= sigma)
    beta_Duration_of_Candidemia_More8Days = pm.Normal('beta_Duration_of_Candidemia_More8Days', mu= mu, sigma= sigma)
    beta_No_Disseminated_Candidiasis = pm.Normal('beta_No_Disseminated_Candidiasis', mu= mu, sigma= sigma)
    beta_Antifungal_Treatment_Amphotericin = pm.Normal('beta_Antifungal_Treatment_Amphotericin', mu= mu, sigma= sigma)
    beta_Antifungal_Treatment_Combination = pm.Normal('beta_Antifungal_Treatment_Combination', mu= mu, sigma= sigma)
    beta_Antifungal_Response_4to7Days = pm.Normal('beta_Antifungal_Response_4to7Days', mu= mu, sigma= sigma)
    beta_Antifungal_Response_More7Days = pm.Normal('beta_Antifungal_Response_More7Days', mu= mu, sigma= sigma)

    #Model fit
    p= pm.Deterministic('p', pm.math.sigmoid(beta_0 + beta_Age*standarize_age + beta_Culture*data.Culture + beta_Fasting_Status*data['Fasting Status']
                                                + beta_Parenteral_Nutrition*data['Parenteral Nutrition'] + beta_Dialysis_During_Candidemia*data['Dialysis During Candidemia']
                                                + beta_Mechanical_Ventilation*data['Mechanical Ventilation'] + beta_Antibiotic*data['Antibiotic (Vancomycin)']
                                                + beta_Broad_Spectrum_Antibiotic*data['Broad-Spectrum Antibiotic']
                                                + beta_Neutropenia*data['Neutropenia'] + beta_Prematurity*data.Prematurity + beta_Malignancy*data.Malignancy 
                                                + beta_Benign_Hematologic_Diseases*data['Benign Hematologic Diseases'] + beta_Heart_Disease*data['Heart Disease']
                                                + beta_Septic_Shock*data['Septic Shock - use of amines'] + beta_Duration_Candidemia_less_2D*data['Duration of Candidemia <= 2 Days']
                                                + beta_Duration_Candidemia_Unknown*data['Duration of Candidemia - Unknown'] + beta_No_SEPSE_qSOFA*data['No SEPSE qSOFA']
                                                + beta_SEPSE_qSOFA*data['SEPSE qSOFA'] + beta_Disseminated_Candidiasis*data['Disseminated Candidiasis'] + beta_Azoles*data['Antifungal Treatment - Azoles']
                                                + beta_Echinocandidae*data['Antifungal Treatment - Echinocandidae'] + beta_Catheter_Not_Removed*data['Catheter Not Removed']
                                                + beta_Catheter_Removed*data['Catheter Removed']+ beta_Catheter_Not_Removed_3Days*data['Catheter Not Removed in the First 3 Days of Candidemia']
                                                + beta_Catheter_Removed_3Days*data['Catheter Removed in the First 3 Days of Candidemia']
                                                + beta_Antifungal_Response_less72H*data['Antifungal Response <72H'] + beta_Antifungal_Response_Unknown*data['Antifungal Response - Unknown']
                                                + beta_Gender*data.Gender + beta_Recent_Abdominal_Surgery*data['Recent Abdominal Surgery (within 30 days)']
                                                + beta_Central_Venous_Device*data['Central Venous Device (CVC, PICC, Hickman, Broviac, Semi-Implantable)']
                                                + beta_Prior_Use_of_Azoles*data['Prior Use of Azoles'] + beta_Previous_Bacteremia*data['Previous Bacteremia (30 days or at candidemia onset)']
                                                + beta_Short_Bowel*data['Short Bowel'] + beta_Inborn_Error_of_Immunity*data['Inborn Error of Immunity'] + beta_Neurological_Sequelae*data['Neurological Sequelae']
                                                + beta_Complete_Antifungal_Treatment_in_days*catd + beta_Duration_of_Candidemia_3to7Days*data['Duration of Candidemia 3-7 Days']
                                                + beta_Duration_of_Candidemia_More8Days*data['Duration of Candidemia >8 Days'] + beta_No_Disseminated_Candidiasis*data['No Disseminated Candidiasis']
                                                + beta_Antifungal_Treatment_Amphotericin*data['Antifungal Treatment - Amphotericin'] + beta_Antifungal_Treatment_Combination*data['Antifungal Treatment - Combination']
                                                + beta_Antifungal_Response_4to7Days*data['Antifungal Response 4-7 Days'] + beta_Antifungal_Response_More7Days*data['Antifungal Response >7 Days']))


    #Likelihood
    pm.Bernoulli('ICU Admissions', p, observed=data.ICU)


# In[103]:


#Sampling
with full_model:
    trace= pm.sample(draws= 2000, tune= 1000, target_accept=0.95, random_seed=42, idata_kwargs={'log_likelihood':True})
summary = az.summary(trace)


# In[167]:


#Print summary
summary.iloc[0:45]


# In[203]:


#Print relevant outcomes
#summary['hdi_3%'][0:45].values
relevant_indexes= []
for i in range(0, len(summary['hdi_3%'][0:45])):
    if not (summary['hdi_3%'][i] < 0) & (summary['hdi_97%'][i] > 0):
        relevant_indexes.append(i)
summary.iloc[[relevant_indexes[0], relevant_indexes[1]],:]


# In[105]:


#Find the Posterior Odds Ratio

#Extract OR
or_samples, indexes=[],[]
for i in summary.iloc[0:45,:].index:
    if not (summary.loc[i,'hdi_3%'] < 0) & (summary.loc[i,'hdi_97%'] > 0):
        #Posterior samples of beta
        or_samples.append(np.exp(summary.loc[i,'mean']))
        #Indexes
        indexes.append(i)

#Find percentage
per=[]
for i in or_samples:
    per.append(round((i - 1)*100, 2))

#Copy dataframe
copy= summary.drop(['sd','mcse_mean','mcse_sd','ess_bulk','ess_tail'], axis=1).loc[indexes,:]

#Insert the new info to the dataframe
copy.insert(1, 'Posterior Odds Ratio', or_samples)
copy.insert(2, 'Percentage', per)

#Print relevant data
copy


# According with the outcome of the Bayesian model, we have the same association compare with the model with relevant features between *Mechanical Ventilation*, *Septic Shock*, and the increase in the probability to be admitted in the ICU.

# In[207]:


#Plot the traces
az.plot_trace(trace, compact=False)


# # Model Assessment

# ### Posterior Predictive Check

# In[212]:


#Sample PPC
with full_model:
    ppc = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=42)

#Plot the PPC
plt.style.use('arviz-colors')
pm.plot_ppc(trace)


# The outcome of the posterior predictions are the same with the last model, indicating that the full model captures the empirical event rate and thus, its considered well balibrated.

# ### Separation Plot

# In[216]:


#Graph the separation plot
az.plot_separation(idata=ppc, y="ICU Admissions",figsize=(12,1))


# The plot shows how well the full model separates the two classes (0 and 1) based on predicted probabilities. From left to right, the bars clearly transition from mostly light blue to mostly dark blue. This means the model assigns higher predicted probabilities to patients who truly had the event/outcome.</br>
# We can observe some dark blue bars mixed into the light zone and vice versa. These represent false negatives and false positives, they are not overly frequent, showing reasonable model performance.

# ### k̂ Parameter

# In[221]:


# compute pointwise LOO
loo = az.loo(trace)

#Plot k values
az.plot_khat(loo.pareto_k)


# On this model we can observe kappa values greater than the model with relevant features, but as mentioned before for a kappa value to be influent the kappa value must be greater than 0.7, in this case all values are under that threshold.

# In[228]:


#Let's find the kappa values greater than 0.6

ax= az.plot_khat(loo.pareto_k.values.ravel()) #ravel puts all the pareto values in one single array
sorted_kappas = np.sort(loo.pareto_k.values.ravel())

#Find the observation where the kappa value exceeds the threshold
threshold= sorted_kappas[-6:].min()
ax.axhline(threshold, ls='--', color= 'gray')
influential_obs= data.reset_index()[loo.pareto_k.values >= threshold].index

for i in influential_obs:
    y= loo.pareto_k.values[i]
    ax.text(i, y + 0.007, str(i), ha= 'center', va='baseline')


# In[262]:


#Take a look on the info about the subjects with the highest kappa value
data.iloc[influential_obs.values,:]
#Alternative
#data.iloc[list(influential_obs.values),:]


# In[274]:


#Print the age of the patients
for i,j in enumerate(data.Age.values[influential_obs.values]):
    year,months=age_converter(j)
    print('Patient %s is %s years and %s months old.' % (i+1,year,months))


# Taking a look on the information of the subjects with a k_hat or kappa value greater than 0.6 come from different ages, it's a group of three wemen and 3 men where four of them where admitted to ICU while only two weren't.

# In[277]:


#Find the subjects on the separation plot

#Create the separation plot
ax= az.plot_separation(idata=ppc, y='ICU Admissions', figsize=(12,1))

#Set the y axis position
y= np.random.uniform(low= 0.1, high= 0.7, size= len(influential_obs))

#Marke the highest kappa observation
for x,y in zip(influential_obs, y):
    #Printing of the value
    text= str(x)
    x= x/len(data)
    #Create the mark
    ax.scatter(x=x, y=y, marker= '+', s=50, color='red', zorder=3)
    ax.text(x=x, y= y + 0.1, s=text, color= 'white', ha='center', va='bottom',
           path_effects= [pe.withStroke(linewidth=2, foreground= 'black')])


# The red crosses mark the observations with the largest κ diagnostics (≈0.6–0.7); none exceed the common influence threshold of 0.7, so none are classified as influential. Subjects 3 and 90 have ICU = 0 (no ICU admission); subjects 4, 77, 79, and 82 have ICU = 1 (admitted). Subjects 3 and 4 are adjacent on the extreme left of the plot, while 90 appears on the right tail; the other marked cases (77, 79, 82) fall in the high predicted-probability region. Because all kappa values are below 0.7, these points do not materially distort estimation — they simply highlight the individual cases with the largest diagnostics (a mix of true positives and two non-events, one of which the model assigns high risk), which reflects expected model uncertainty rather than model failure.

# ### Forest Plot

# In[291]:


#Graph the Forest plot
az.plot_forest(trace, combined=True, colors= 'slategray', var_names=summary.index[0:45], figsize=(6,13.5))
plt.axvline(x=0, c= 'red', linestyle='--',alpha = 0.4)


# The forest plot validates the outcomes from summary, the only two features that do not contain zero on their credible interval are *Mechanical Ventilation* and *Septic Shock*, both of them are on the right side of the forest plot indicating these features increase the odds of being admitted to the ICU.

# ### Confusion Matrix

# In[296]:


#Posterior-mean probability for each patient
y_prob = trace.posterior['p'].mean(('chain','draw')).values

#Bayes optimal hard call
#Element-wise threshold at 0.5 -> True/False.
y_pred= (y_prob > 0.5).astype(int)

#Confussion matrix
print('Confusion Matrix \n',confusion_matrix(data.ICU, y_pred))
print(classification_report(data.ICU, y_pred, digits=3))


# The model correctly classified 28 of 36 non-ICU cases (specificity = 77.8%) and 58 of 62 ICU admissions (sensitivity = 93.5%), indicating a strong ability to identify patients who required ICU care.</br>
# 
# Precision was high for both classes (0.875 for non-ICU and 0.879 for ICU), showing that the predicted labels closely matched the true outcomes. The F1-scores (0.824 for non-ICU and 0.906 for ICU) further confirm balanced performance between precision and recall, particularly for the ICU class, where accurate identification is clinically most relevant.</br>
# 
# Overall accuracy reached 87.8%, and the macro-averaged metrics (precision = 0.877, recall = 0.857, F1 = 0.865) demonstrate that the model performs consistently across both outcome categories. These results indicate that incorporating all features improves discrimination and yields a robust predictive model for ICU admission in pediatric candidemia cases.</br>

# ### Area Under the Curve

# In[300]:


#Create a dictionary to get the parameters for AUC
pred_scores= dict(y_true=data.ICU, y_score=y_pred)
#Print AUC
print(f'AUC = {roc_auc_score(**pred_scores):.4f}')


# The model achieved an Area Under the ROC Curve (AUC) of 0.8566, indicating strong overall discriminative ability. An AUC above 0.85 reflects that the model correctly ranks ICU and non-ICU cases with high reliability, further supporting the robustness of the predictive performance obtained with all features included.

# # Compare models with LOO-PSIS

# This section is dedicated to compare both model to determine which one has a better perfomance, it's intended to use the method Leave One Out using Pareto-smoothed importance sampling (PSIS).

# In[32]:


#Compute LOO-PSIS
loo_reduced= az.loo(idata, pointwise=True)
loo_full= az.loo(trace, pointwise=True)

#Compare both models
az.compare({'reduced': loo_reduced, 'full': loo_full},
           ic='loo', scale='log')


# Before digging into the explanation on how to interpret the outcome, it is vital to highlight that the first model fitted on this notebook, the one that is considering the only the relevant features, is called *reduced* for this analysis, and the model considering all the features is called *full*.</br> 
# 
# The interpertation of the outcome to determine which model is better is straight forward, the function *az.compare* places the model with the best performance in the first row. To validate this information and understand the procedure behind this function, we have to take into account the parameters *elpd_loo*, *elpd_diff*, and *weight*. </br>
# 
# The interpretation of the *elpd_diff* is based on the highest value, or in this case the less negative is the better model with a better predictive accuracy. In this case the reduced model has the less negative value.</br>
# 
# *Elpd_diff* tells if the difference between the models is meaningful, it is calculated as *elpd_loo(reduced)−elpd_loo(full)*, the results shows a difference moderately large in favor of the reduced model, following the interpretation scale (used in the LOO literature, e.g. Vehtari et al. 2017):</br>
# 
# - elpd_diff   ---         Interpretation
# - 0–1	 --------        negligible
# - 1–3	 --------          small
# - 3–7	 --------          moderate
# - +7	 ---------          large</br>
# 
# *P_loo* tells you how complex the model behaves in terms of overfitting and flexibility, not just how many raw parameters it has. The interpretation is:
# - Higher p_loo -> more flexible model -> higher risk of overfitting.
# - Lower p_loo -> simpler, more regularized model.</br>
# The reduced model has lower effective complexity, generalizes better.</br>
# 
# In conclusion, the PSIS-LOO comparison indicates that the reduced model provides better out-of-sample predictive performance than the full model.

# # Conclusion

# Across both Bayesian logistic regression models, *mechanical ventilation* and *septic shock* consistently emerged as the only variables strongly associated with ICU admission. In both forest plots, their posterior distributions were positioned clearly to the right, and their credible intervals did not cross zero, indicating a robust and clinically meaningful increase in the odds of ICU admission for patients presenting with either condition.</br>
# 
# The predictive performance of the models further supports these findings. Both versions achieved high accuracy, balanced precision and recall, and strong discriminative ability, with AUC values above 0.80 (0.8047 for the reduced model and 0.8566 for the full model). The confusion matrices and classification reports show particularly strong sensitivity for detecting ICU admissions—an important feature in clinical prediction—while maintaining good specificity for identifying non-ICU cases.</br>
# 
# Taken together, the results indicate that although the dataset contains many potential predictors, ICU admission is primarily driven by acute clinical severity indicators rather than demographic or chronic disease factors. The models demonstrate stable performance and provide consistent evidence that *mechanical ventilation* and *septic shock* are the key determinants of ICU requirement in this patient population.

# In[308]:


#This is a LOO-PIT assessment, in this case does not apply because is for continous outcomes, is recommended for a Bayesian linear regression
#with relevant_feat_model:
#    ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

#plt.style.use('arviz-colors')
#pm.plot_loo_pit(idata,  # using arviz InfereceData type 
#                y='ICU Admissions', 
#                color='slategray')
#plt.show()

