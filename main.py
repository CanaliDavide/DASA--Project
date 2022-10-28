from audioop import avg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn as sk
import scipy

np.random.seed(0)

chickenData = pd.read_excel("Data/dataset.xlsx")
chickenData["Chickens Death Per Day"] = chickenData["Chickens Death Per Day"].fillna(chickenData["Chickens Death Per Day"].mean())
chickenData["Current Chickens"] = chickenData["Current Chickens"].fillna(chickenData["Current Chickens"].mean())
chickenData["# Eggs sold (First quality)"] = chickenData["# Eggs sold (First quality)"].fillna(chickenData["# Eggs sold (First quality)"].mean())
chickenData["Water Consuption (gr)"] = chickenData["Water Consuption (gr)"].fillna(chickenData["Water Consuption (gr)"].mean())
chickenData["Date of Selling"] = chickenData["Date of Laid"]
print(chickenData.head())
print(chickenData.shape)

""" chickenData2 = chickenData[["Chickens Death Per Day", "Water Consuption (gr)", "Feed Consuption (gr)"]]
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(chickenData2)
plt.show() """

#Parameters choice
x_feautures = ["Water Consuption (gr)", "Feed Consuption (gr)", "Current Chickens", "Chickens Death Per Day"]
x = chickenData[x_feautures].to_numpy()
#NB: y must be monodimensional
y = chickenData["# Eggs sold (First quality)"].to_numpy()
print(x.shape, y.shape)

#model split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=23)

# Step 1: add x0=1 to the dataset
x_train_0 = np.c_[np.ones((x_train.shape[0],1)),x_train]
x_test_0 = np.c_[np.ones((x_test.shape[0],1)),x_test]

# Step2: build the model
theta = np.matmul(np.linalg.inv(np.matmul(x_train_0.T,x_train_0) ), np.matmul(x_train_0.T,y_train)) 

# we now create a pandas data frame to compare the linear regression models parameters
parameter = ['theta_'+str(i) for i in range(x_train_0.shape[1])]
columns = ['intercept'] + list(chickenData[x_feautures].columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

# Note: there is no need to add x_0=1, sklearn will take care of it when fit_intercept equals True
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression(fit_intercept=True)   
lin_reg.fit(x_train, y_train)    

# You can inspect the learned parameters by using the 'intercept_' and 'coef_' attributes of the model
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameters = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
#print(parameters)




#MODEL VALUATION


# Normal equation evaluation

y_pred_norm =  np.matmul(x_test_0,theta)

# MSE
mse = np.sum((y_pred_norm - y_test)**2) / x_test_0.shape[0]

# R_squared 
rss = np.sum((y_pred_norm - y_test)**2)
tss = np.sum((y_test - y_test.mean())**2)
R_squared = 1 - (rss/tss)
print('The Mean Squared Error (MSE) is: ', mse)
print('R squared obtained from the normal equation method is:', R_squared)

# sklearn regression module evaluation

y_pred_sk = lin_reg.predict(x_test)

# MSE
from sklearn.metrics import mean_squared_error
mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_squared
R_squared_sk = lin_reg.score(x_test,y_test)
print('The Mean Squared Error (MSE) is: ', mse_sk)
print('R squared obtained from scikit learn library is :',R_squared_sk)


# Check for Linearity
residuals = y_test - y_pred_sk


# Compute the residuals on the test set
""" plt.figure(figsize=(5,5))
sns.scatterplot(x=y_pred_sk, y=residuals, color='r')
plt.title('Check for Linearity:\nResiduals vs Predicted values')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show() """


# Check for residuals normality & mean
""" sns.displot(data=residuals, kind='hist', kde=True)
plt.axvline(residuals.mean(), color='k', linestyle='--')
plt.title('Check for residuals normality & mean:\n Distribution of residuals');
plt.xlabel('Residuals')
plt.show() """

# Plotting the correlation matrix 
sns.heatmap(chickenData[x_feautures].corr(), annot=True)
plt.title('Correlation of Predictors')
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Gathering the VIF for each variable
VIF = [variance_inflation_factor(chickenData[x_feautures], i) for i in range(chickenData[x_feautures].shape[1])]
for idx, vif in enumerate(VIF):
    print('{0}: {1}'.format(x_feautures[idx], vif))