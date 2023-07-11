import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('D:\Projects\ML\Wine Quality Prediction ML model\winequalityN.csv')
print(df.head(2))

#types of data present in each column of the dataset
print(df.info())

#descriptive statistical measures of the dataset.
print(df.describe().T)
