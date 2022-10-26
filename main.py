import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import scipy

np.random.seed(0)

chickenData = pd.read_excel("Data/dataset.xlsx")
chickenData["Chickens Death Per Day"] = chickenData["Chickens Death Per Day"].fillna(0)
chickenData["Current Chickens"] = chickenData["Current Chickens"].fillna(0)
chickenData["# Eggs sold (First quality)"] = chickenData["# Eggs sold (First quality)"].fillna(0)
chickenData["Date of Selling"] = chickenData["Date of Laid"]
chickenData = chickenData[["Chickens Death Per Day", "Water Consuption (gr)", "Feed Consuption (gr)"]]
print(chickenData.head())
print(chickenData.shape)
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(chickenData)
plt.show()