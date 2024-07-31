'''Import the necessary modules for analysis'''
import numpy as np # Numerical calculations
import pandas as pd # Data handling
load = pd.read_csv("C:/Users/OWNER/OneDrive/Documents/Python/Lesson 2/lcfs_2013.csv")

dta = load.head() # Show the first few observations
print(dta)

ldescr = load.describe() #Descriptive Statistics
print(ldescr)

'''A good way to visualise a dataset before running any analysis is to use a scatter plot using the
matplotlib and seaborn modules'''

import matplotlib.pyplot as plt # Lower-level graphics
import seaborn as sns # High-level graphics

# Plot a scatter diagram of the 'lcfs' data
'''The dataset has outliers because this variable has been ‘top-coded’ (income above £1184.99 a week is 
censored and recorded as £1184.99 -regardless of actual income and expenditure above £1144.48 is recorded
as £1144.48 regardless of the actual expenditure). I am chossing to remove these high-earners from the 
sample and re-analyse the data to see if that changes the distribution of income.'''
#Filter data to remove outliers from high-earners
inc_threshold = 1184.99 # define the income threshold
exp_threshold = 1144.48 # define the expenditure threshold

new = load[load['P344pr'] < inc_threshold] # Filter the dataset to remove income outliers
new2 = new[new["P550tpr"] < exp_threshold] # Filter the dataset to remove expenditure outliers

#I was curious to see how the descriptive statistics change after removing the income and expediture outliers.
ndescr = new2.describe() #Descriptive Statistics
print(ndescr)
'''About 869 observations were removed.'''

# Extract the relevant columns
x = new2['P344pr'] # Income
y = new2['P550tpr'] # Expendiiture

#create a scatter plot
plt.scatter(x,y, c='red', s=20, alpha=0.5)

#Add titles and labels
plt.title('Income and Expenditure Scatter Plot')
plt.xlabel('Income')
plt.ylabel('Expenditure')

#show the plot
plt.show()

# Estimating a linear regression model
import statsmodels.formula.api as smf # Econometrics
import scipy.stats as stats # Statistics
mod1 = smf.ols('P550tpr ~ P344pr', data= load).fit()
b1 = mod1.params['Intercept']
b2 = mod1.params['P344pr']
smod1 = mod1.summary()
print(smod1)

# Collect regression results in a table and print it
# First, we make a "dictionary;" then, we make it a dataframe:
tbl = {'coeff': mod1.params, 'std.err': mod1.bse,
       't-value': mod1.tvalues, 'p-value': mod1.pvalues}
df1 = pd.DataFrame(tbl)
print(df1.round(4))

# Plot the fitted regression plot
# Extract the relevant columns for the scatter plot
x = load['P344pr'] #Income
y = load['P550tpr'] #Expenditure

# Plot the original data
plt.scatter(x,y, c='grey', s=10, alpha=0.1)

# Plot the fitted regression line
# Generate the fitted values
x_fit = pd.DataFrame({'P344pr': x})
y_fit = mod1.predict(x_fit)

plt.plot(x, y_fit, c= 'red')
# Add titles and labels
plt.title('Scatter Diagram with Fitted Regression Line for the inc_exp model')
plt.xlabel('Income')
plt.ylabel('Expenditure')
plt.legend()

# Show the plot
plt.show()
