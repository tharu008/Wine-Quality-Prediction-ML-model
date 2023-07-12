import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('D:\Projects\ML\Wine Quality Prediction ML model\winequalityN.csv')
#print(df.head(2))

# Let’s inpute the missing values by means as the data present in the different columns are continuous values.
for col in df.columns:
    if df[col].isnull().sum() > 0:
        # fillna() is a method in pandas to fill the missing values with a specific value (here it is mean)
        df[col] = df[col].fillna(df[col].mean())

# overall total sum of empty columns
df.isnull().sum().sum()

# Let’s draw the histogram to visualise the distribution of the data with continuous values in the columns of the dataset.
df.hist(bins=20, figsize=(10, 10))
plt.show()
