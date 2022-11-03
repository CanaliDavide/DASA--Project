from audioop import avg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn as sk
import scipy

# data load
cycleA = pd.read_csv("Data/CycleA-Data301022.csv", sep=";")
cycleB = pd.read_csv("Data/CycleB-Data301022.csv", sep=";")

# fill NaN value
cycleA["Chickens Death Per Day"] = cycleA["Chickens Death Per Day"].fillna(0)
cycleA["Current Chickens"] = cycleA["Current Chickens"].fillna(0)
cycleA["# of Eggs"] = cycleA["# of Eggs"].fillna(0)
cycleB["Chickens Death Per Day"] = cycleA["Chickens Death Per Day"].fillna(0)
cycleB["Current Chickens"] = cycleA["Current Chickens"].fillna(0)
cycleB["# of Eggs"] = cycleA["# of Eggs"].fillna(0)

# data preview
print("A")
print(cycleA.head())
print("B")
print(cycleB.head())

# input (x) and output (y) feautures
x_feautures = ["Water Consuption (gr)", "Feed Consuption (gr)"]
y_feautures = ["# Eggs sold (First quality)", "Current Chickens", "Chickens Death Per Day", "# of Eggs"]

# pair plot cycle A
sns.set(style="ticks", color_codes=True)
sns.pairplot(cycleA[x_feautures + y_feautures])
plt.show()

# heatmap cycle A
sns.heatmap(cycleA[x_feautures + y_feautures].corr(), annot=True)
plt.show()
# found a strong correlation betwen:
#   -   FEED and EGGS
#   -   WATER and DEATH

# pair plot cycle B
sns.set(style="ticks", color_codes=True)
sns.pairplot(cycleB[x_feautures + y_feautures])
plt.show()

# heatmap cycle B
sns.heatmap(cycleB[x_feautures + y_feautures].corr(), annot=True)
plt.show()
# found a strong correlation betwen:
#   -   FEED and EGGS
#   -   WATER and DEATH
#   -   #EGGS and #EGGS SOLD --> obv


# because of the correlations found (that are similar in both cycles), it may be convenient to create a model to predect:
#   -   number of deaths in relation to the water given to the chickens (and maybe the wather)
#   -   number of eggs produced in relation to the feed