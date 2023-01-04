# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:07:05 2022

@author: vaishnav
"""

#hypothesis testing
#Buyer ratio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats
from scipy.stats import norm


df=pd.read_csv(r'Downloads\BuyerRatio.csv')
df

sns.distplot(df["East"],color=('blue'))

sns.distplot(df["West"],color=('red'))

sns.distplot(df["North"],color=('yellow'))

sns.distplot(df["South"],color=('purple'))


d1=df.drop(['Observed Values'],axis=1)
d1

from scipy.stats import chi2_contingency
obs=np.array([[50,142,131,70],[435,1523,1356,750]])
obs

# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)

# Compare p_value with α = 0.05
#Inference: As (p-value = 0.6603) > (α = 0.05); 
#Accept the Null Hypothesis i.e. Independence of categorical variables Thus,
# male-female buyer rations are similar across regions and are not related


#=======================================================================================================================================================================================================================================================================
#Costomer.csv

order = pd.read_csv(r"Downloads\Costomer+OrderForm.csv")
order


for i in order.columns:
    print(order[i].value_counts())
    print()
    
    
X = ['Philippines','Indonesia','malta','India']
ErrorFree = [271,267,269,280]
Defective= [29,33,31,20]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, ErrorFree , 0.4, label = 'ErrorFree')
plt.bar(X_axis + 0.2, Defective, 0.4, label = 'Defective')
plt.xticks(X_axis, X)
plt.legend()
plt.show()


# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# Compare p_value with α = 0.05
#Inference: As (p_value = 0.2771) > (α = 0.05); 
#Accept Null Hypthesis i.e. Independence of categorical variables Thus,
#customer order forms defective % does not varies by centre



#=======================================================================================================================================================================================================================================================================
#Cutlets.csv

cutlet = pd.read_csv(r"Downloads\Cutlets.csv")
cutlet

sns.distplot(cutlet["Unit A"],color="Red")

sns.distplot(cutlet["Unit B"],color="Black")

sns.distplot(cutlet["Unit A"],color="Red")
sns.distplot(cutlet["Unit B"],color="green")
plt.legend(['Unit A','Unit B'])




stats.ttest_rel(cutlet["Unit A"], cutlet["Unit B"])


stats.ttest_ind(cutlet["Unit A"], cutlet["Unit B"])

stats.shapiro(cutlet["Unit A"])

stats.shapiro(cutlet["Unit B"])


#=======================================================================================================================================================================================================================================================================
#LabTaT.csv


lab = pd.read_csv(r"Downloads\LabTAT.csv")
lab

lab.mean()

sns.distplot(lab["Laboratory 1"],color="Red")

sns.distplot(lab["Laboratory 2"],color="Black")

sns.distplot(lab["Laboratory 3"],color="purple")

sns.distplot(lab["Laboratory 4"],color="green")


stats.f_oneway(lab.iloc[:,0], lab.iloc[:,1],lab.iloc[:,2],lab.iloc[:,3])


#Final Statement : as we can see above pvalue = 2.11 x 10 raise to -57 which is almost 0 and lesser than alpha value hence we reject H0 (Null hypothesis)
#pvalue < alpha ( 2.11 x 10 raise to -57 < 0.05)
#Accept Ha => At least 1 Lab's Average TAT is different // Not all the averages are same



















